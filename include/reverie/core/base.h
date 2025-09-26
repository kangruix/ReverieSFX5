#pragma once
#include "reverie/core/device.h"
#include "reverie/core/buffer.h"
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

namespace reverie {

class ReverieBase : public nanobind::intrusive_base {
public:
	virtual ~ReverieBase() = default;

	const Device& device() const { return m_device; }

	virtual std::string to_string() const;

protected:
	Device m_device;
};

using nanobind::ref;

} // namespace reverie