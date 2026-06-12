add_library(ray_tracing
  "systems/ray_tracer.cpp"
  )
target_link_libraries(ray_tracing Aperture)

add_executable(raytrace
  "main.cpp"
  )
target_link_libraries(raytrace ray_tracing Aperture)
add_dependencies(problems raytrace)
