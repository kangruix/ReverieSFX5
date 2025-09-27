#pragma once
#include "reverie/core/device.h"
#include "reverie/core/buffer.h"
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

namespace reverie {

/**
 * \brief Reverie base class with intrusive reference counting.
 * 
 * The use of nanobind::intrusive_base (along with nanobind::ref)
 * enables an efficient alternative to std::shared_ptr that can
 * be seamlessly exchanged across C++ and Python.
 *
 * See https://nanobind.readthedocs.io/en/latest/ownership_adv.html
 */
class ReverieBase : public nanobind::intrusive_base {
public:
	/// Constructor
	ReverieBase(const Device& device) : m_device(device) {}
	
	/// Virtual destructor
	virtual ~ReverieBase() = default;

	const Device& device() const { return m_device; }

	virtual std::string to_string() const;

protected:
	Device m_device;
};

using nanobind::ref;

} // namespace reverie