#pragma once
#include "reverie/core/geometry/voxelgrid.h"

namespace reverie {
namespace fdtd {

/// FDTD configuration parameters
struct Config {
    float cellsize = 0.005;
    int samplerate = 120000;

    float sound_speed = 343;
    float air_density = 1.2;
};

class Grid : public geometry::VoxelGrid {
public:
    float* d_p1 = nullptr;      ///< current pressure
    float* d_p0 = nullptr;      ///< previous pressure

    uint8_t* d_md = nullptr;    ///< cell metadata
    int step = 0;               ///< current time step

    const double dt;            ///< timestep size
    const float lambda;         ///< Courant number (C * dt / dx)

    /// Constructor: initialize from global voxel-space coordinates
    Grid(point3i vmin, point3i vmax, Config conf, 
        int step = 0, Device device = Device());

    /// Destructor
    ~Grid();

    /// Advances this Grid by \c num_steps
    void solve(int num_steps = 1);

    std::string to_string() const override;

private:
    Config m_config;

    Buffer<float> m_pressure;
    Buffer<uint8_t> m_metadata;

    //std::vector<ref<Object>> m_objects;
    //std::vector<ref<Emitter>> m_sources;
    //std::vector<ref<Probe>> m_receivers;
};

} // namespace fdtd
} // namespace reverie