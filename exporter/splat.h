#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

struct GaussianCPU {
  glm::vec3 mean;
  float _pad0;

  glm::vec3 scale;
  float _pad1;

  glm::vec4 rotation;

  glm::vec3 color;
  float opacity;
};

struct GaussianGPU {
  glm::vec2 meanxy;
  glm::vec2 meanz_color;
  float cov3d[6];
}; // 40bytes