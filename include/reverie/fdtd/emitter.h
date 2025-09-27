#pragma once

namespace reverie {
namespace fdtd {

// Forward declarations
class Grid;

class Emitter {
public:
    Emitter(const Grid* grid);

    void inject();

private:
    const Grid* m_grid = nullptr;
};

} // namespace fdtd
} // namespace reverie