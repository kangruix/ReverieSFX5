#include "reverie/core/base.h"
#include <nanobind/intrusive/counter.inl>

namespace reverie {

std::string ReverieBase::to_string() const {
	return m_device.to_string();
}
    
} // namespace reverie