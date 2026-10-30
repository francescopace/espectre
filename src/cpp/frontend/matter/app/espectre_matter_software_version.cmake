# ESPectre Matter software version
#
# Matter reports SoftwareVersion as a uint32 that must grow with every release.
# esp-matter takes it from PROJECT_VER_NUMBER. This encodes the git-describe
# identity in the same order as `compare_ota_versions()` in
# `src/cpp/frontend/ota_version.cpp`:
#
#   major:5 | minor:6 | patch:5 | stage:4 | distance:12
#
# `stage` is N for an `rcN` prerelease (1-9) and 15 for a final release;
# `distance` is the git-describe commit count. An identity that does not fit,
# such as the local `0.0.0` fallback or another prerelease label, encodes as 0.

function(espectre_matter_software_version identity out_var)
    set(_value 0)
    if(identity MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(-rc([1-9]))?(-([0-9]+)-g[0-9a-f]+)?$")
        set(_major "${CMAKE_MATCH_1}")
        set(_minor "${CMAKE_MATCH_2}")
        set(_patch "${CMAKE_MATCH_3}")
        set(_stage 15)
        if(NOT "${CMAKE_MATCH_5}" STREQUAL "")
            set(_stage "${CMAKE_MATCH_5}")
        endif()
        set(_distance 0)
        if(NOT "${CMAKE_MATCH_7}" STREQUAL "")
            set(_distance "${CMAKE_MATCH_7}")
        endif()
        if(_major LESS 32 AND _minor LESS 64 AND _patch LESS 32 AND _distance LESS 4096
           AND NOT (_major EQUAL 0 AND _minor EQUAL 0 AND _patch EQUAL 0))
            math(EXPR _value
                "(${_major} << 27) | (${_minor} << 21) | (${_patch} << 16) | (${_stage} << 12) | ${_distance}")
        endif()
    endif()
    if(_value EQUAL 0)
        message(WARNING "Matter SoftwareVersion is 0: '${identity}' cannot be encoded")
    endif()
    set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()
