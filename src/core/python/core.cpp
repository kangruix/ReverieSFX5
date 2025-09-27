#include <nanobind/nanobind.h>
#include "reverie/core/base.h"

namespace nb = nanobind;
using namespace reverie;

NB_MODULE(core, m) {
    nb::intrusive_init(
        [](PyObject* o) noexcept { nb::gil_scoped_acquire guard; Py_INCREF(o); },
        [](PyObject* o) noexcept { nb::gil_scoped_acquire guard; Py_DECREF(o); });

    nb::class_<ReverieBase>(m, "ReverieBase",
        nb::intrusive_ptr<ReverieBase>(
            [](ReverieBase* rb, PyObject* po) noexcept { rb->set_self_py(po); }));
}