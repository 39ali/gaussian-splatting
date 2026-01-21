
#include "splat.h"
#include <bit>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct PropertyInfo {
  std::string name;
  std::string type;
};

struct PlyHeader {
  bool binary = false;
  size_t vertexCount = 0;
  std::vector<PropertyInfo> properties;
};

PlyHeader parsePlyHeader(std::ifstream &file) {
  PlyHeader header;
  std::string line;
  while (std::getline(file, line)) {
    if (line == "end_header")
      break;
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    if (token == "format") {
      std::string fmt;
      iss >> fmt;
      header.binary = (fmt.find("binary") != std::string::npos);
    } else if (token == "element") {
      std::string elem;
      iss >> elem;
      if (elem == "vertex") {
        iss >> header.vertexCount;
      }
    } else if (token == "property") {
      std::string type, name;
      iss >> type >> name;
      header.properties.push_back({name, type});
    }
  }
  return header;
}

inline float SH_C0 = 0.28209479177387814f;

glm::vec3 shToRGB(float sh0, float sh1, float sh2) {
  float r = 0.5f + SH_C0 * sh0;
  float g = 0.5f + SH_C0 * sh1;
  float b = 0.5f + SH_C0 * sh2;
  return glm::vec3(glm::clamp(r, 0.0f, 1.0f), glm::clamp(g, 0.0f, 1.0f),
                   glm::clamp(b, 0.0f, 1.0f));
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<GaussianCPU> loadGaussianPLY(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open PLY: " + path);

  PlyHeader header = parsePlyHeader(file);
  if (!header.binary)
    throw std::runtime_error("ASCII PLY not supported yet");

  std::vector<GaussianCPU> gaussians(header.vertexCount);

  for (size_t i = 0; i < header.vertexCount; i++) {
    GaussianCPU g{};
    float sh_dc[3] = {0, 0, 0};

    for (const auto &prop : header.properties) {
      float v;

      // Read based on type
      if (prop.type == "float") {
        file.read(reinterpret_cast<char *>(&v), sizeof(float));
      } else if (prop.type == "double") {
        double d;
        file.read(reinterpret_cast<char *>(&d), sizeof(double));
        v = static_cast<float>(d);
      } else if (prop.type == "uchar") {
        unsigned char uc;
        file.read(reinterpret_cast<char *>(&uc), sizeof(unsigned char));
        v = static_cast<float>(uc);
      } else {
        // Default to float for unknown types
        file.read(reinterpret_cast<char *>(&v), sizeof(float));
      }

      // Parse property
      if (prop.name == "x")
        g.mean.x = v;
      else if (prop.name == "y")
        g.mean.y = v;
      else if (prop.name == "z")
        g.mean.z = v;
      else if (prop.name == "f_dc_0")
        sh_dc[0] = v;
      else if (prop.name == "f_dc_1")
        sh_dc[1] = v;
      else if (prop.name == "f_dc_2")
        sh_dc[2] = v;
      else if (prop.name == "opacity")
        g.opacity = sigmoid(v); // Convert from logit space
      else if (prop.name == "scale_0")
        g.scale.x = std::exp(v); // Convert from log space
      else if (prop.name == "scale_1")
        g.scale.y = std::exp(v);
      else if (prop.name == "scale_2")
        g.scale.z = std::exp(v);
      else if (prop.name == "rot_0")
        g.rotation.w = v;
      else if (prop.name == "rot_1")
        g.rotation.x = v;
      else if (prop.name == "rot_2")
        g.rotation.y = v;
      else if (prop.name == "rot_3")
        g.rotation.z = v;
    }

    // Convert SH to RGB
    g.color = shToRGB(sh_dc[0], sh_dc[1], sh_dc[2]);

    gaussians[i] = g;
  }

  return gaussians;
}

std::vector<GaussianGPU>
preprocessGaussians(const std::vector<GaussianCPU> &cpuGaussians) {
  std::vector<GaussianGPU> gpuGaussians;
  gpuGaussians.reserve(cpuGaussians.size());

  for (size_t i = 0; i < cpuGaussians.size(); ++i) {
    const auto &g = cpuGaussians[i];

    GaussianGPU gpu;
    gpu.meanxy.x = g.mean.x;
    gpu.meanxy.y = g.mean.y;
    gpu.meanz_color.x = g.mean.z;

    uint32_t r = round(g.color.x * 255.0);
    uint32_t green = round(g.color.y * 255.0);
    uint32_t b = round(g.color.z * 255.0);
    uint32_t a = round(g.opacity * 255.0);
    uint32_t packed = (r) | (green << 8) | (b << 16) | (a << 24);
    gpu.meanz_color.y = std::bit_cast<float>(packed);

    // Build rotation matrix from quaternion
    glm::quat q = glm::normalize(
        glm::quat(g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z));
    glm::mat3 R = glm::mat3_cast(q);

    // Scale matrix
    glm::mat3 S = glm::mat3(g.scale.x, 0.0f, 0.0f, 0.0f, g.scale.y, 0.0f, 0.0f,
                            0.0f, g.scale.z);

    // Compute M = R * S
    glm::mat3 M = R * S;

    // Compute 3D covariance: Σ = M * M^T
    glm::mat3 Sigma = M * glm::transpose(M);

    // Store upper triangle (symmetric matrix)
    gpu.cov3d[0] = Sigma[0][0]; // Σ00
    gpu.cov3d[1] = Sigma[1][0]; // Σ01 (was Sigma[0][1])
    gpu.cov3d[2] = Sigma[2][0]; // Σ02 (was Sigma[0][2])
    gpu.cov3d[3] = Sigma[1][1]; // Σ11
    gpu.cov3d[4] = Sigma[2][1]; // Σ12 (was Sigma[1][2])
    gpu.cov3d[5] = Sigma[2][2];

    gpuGaussians.push_back(gpu);
  }

  return gpuGaussians;
}

void saveGaussianGPU(const std::string &path,
                     const std::vector<GaussianGPU> &gaussians) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Failed to open output file");

  uint64_t count = gaussians.size();
  out.write(reinterpret_cast<const char *>(&count), sizeof(count));
  out.write(reinterpret_cast<const char *>(gaussians.data()),
            sizeof(GaussianGPU) * count);
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input.ply> <output.bin>\n";
    return 1;
  }

  const std::string inputPath = argv[1];
  const std::string outputPath = argv[2];

  auto gaussians = loadGaussianPLY(inputPath);

  auto gpuGaussian = preprocessGaussians(gaussians);

  saveGaussianGPU(outputPath, gpuGaussian);

  std::cout << "Saved " << gpuGaussian.size()
            << " GaussianGPU entries to gaussians_gpu.bin\n";

  return 0;
}
