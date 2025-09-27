#include "reverie/core/buffer.h"
#include "reverie/core/vector.h"
#include "reverie/core/backend/memory.h"

namespace reverie {

template <typename T>
Buffer<T>::Buffer(size_t size, const Device& dev)
	: m_size(size), m_device(dev) {
	allocate();
}

template <typename T>
Buffer<T>::Buffer(const std::vector<T>& vec, const Device& dev)
	: m_size(vec.size()), m_device(dev) {
	allocate();
	copy_from(vec.data(), DeviceType::CPU);
}

template <typename T>
Buffer<T>::Buffer(const Buffer<T>& other, const Device& dev)
	: m_size(other.m_size), m_device(dev) {
	allocate();
	copy_from(other.data(), other.m_device.type);
}

template <typename T>
Buffer<T>::Buffer(Buffer<T>&& other) noexcept
	: m_data(other.m_data), m_size(other.m_size), 
	  m_device(other.m_device) {
	other.m_data = nullptr;
	other.m_size = 0;
}

template <typename T>
Buffer<T>& Buffer<T>::operator=(Buffer<T>&& other) noexcept {
	if (this != &other) {
		deallocate();
		
		m_data = other.m_data;
		m_size = other.m_size;
		m_device = other.m_device;
		
		other.m_data = nullptr;
		other.m_size = 0;
	}
	return *this;
}

template <typename T>
Buffer<T>::~Buffer() {
	deallocate();
}

template <typename T>
void Buffer<T>::fill(int value) {
	if (m_device.type == DeviceType::CPU) {
		std::memset(m_data, value, m_size * sizeof(T));
	}
#ifdef REV_ENABLE_CUDA
	else if (m_device.type == DeviceType::CUDA) {
		cudaCheck(cudaMemset(m_data, value, m_size * sizeof(T)));
	}
#endif
}

template <typename T>
const T* Buffer<T>::view() const {
	if (m_device.type == DeviceType::CPU) {
		return m_host_data;
	}
#ifdef REV_ENABLE_CUDA
	else if (m_device.type == DeviceType::CUDA) {
		cudaCheck(cudaMemcpy(m_host_data, m_data, m_size * sizeof(T), DeviceToHost));
		return m_host_data;
	}
#endif
}

template <typename T>
void Buffer<T>::allocate() {
	if (m_size > 0) cudaCheck(cudaMalloc((void**) &m_data, m_size * sizeof(T)));
	
	if (m_device.type == DeviceType::CPU) {
		m_host_data = m_data;
	}
#ifdef REV_ENABLE_CUDA
	else if (m_device.type == DeviceType::CUDA) {
		m_host_data = new T[m_size];
	}
#endif
}

template <typename T>
void Buffer<T>::deallocate() {
	if (m_size > 0) cudaCheck(cudaFree(m_data));

#ifdef REV_ENABLE_CUDA
	if (m_device.type == DeviceType::CUDA && m_host_data) {
		delete[] m_host_data;
		m_host_data = nullptr;
	}
#endif
}

template <typename T>
void Buffer<T>::copy_from(const T* src_ptr, DeviceType src_dev) {
	if (m_device.type == DeviceType::CPU && src_dev == DeviceType::CPU) {
		std::memcpy(m_data, src_ptr, m_size * sizeof(T));
	}
#ifdef REV_ENABLE_CUDA
	else if (m_device.type == DeviceType::CUDA && src_dev == DeviceType::CUDA) {
		cudaCheck(cudaMemcpy(m_data, src_ptr, m_size * sizeof(T), DeviceToDevice));
	}
	else if (m_device.type == DeviceType::CPU && src_dev == DeviceType::CUDA) {
		cudaCheck(cudaMemcpy(m_data, src_ptr, m_size * sizeof(T), DeviceToHost));
	}
	else if (m_device.type == DeviceType::CUDA && src_dev == DeviceType::CPU) {
		cudaCheck(cudaMemcpy(m_data, src_ptr, m_size * sizeof(T), HostToDevice));
	}
#endif
}

template class Buffer<int>;
template class Buffer<float>;

template class Buffer<point3f>;
template class Buffer<vec3i>;

} // namespace reverie