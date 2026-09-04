option(ESPECTRE_ENABLE_COVERAGE "Enable compiler coverage flags for host-side tests" OFF)

function(espectre_apply_coverage target_name)
    if(NOT ESPECTRE_ENABLE_COVERAGE)
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options("${target_name}" PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options("${target_name}" PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options("${target_name}" PRIVATE --coverage)
        # Model the GCC runtime as a library so CMake places it after any
        # instrumented static libraries that introduce __gcov_* references.
        target_link_libraries("${target_name}" PRIVATE gcov)
    endif()
endfunction()
