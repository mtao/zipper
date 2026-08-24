#pragma once

#include <string>

namespace zipper::utils::solver {

struct SolverError {
    enum class Kind {
        diverged,
        breakdown,
        invalid_input,
    };

    Kind kind;
    std::string message;
};

} // namespace zipper::utils::solver
