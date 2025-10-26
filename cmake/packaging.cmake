# CPack Configuration for Vesper

set(CPACK_PACKAGE_NAME "vesper")
set(CPACK_PACKAGE_VENDOR "Vesper Contributors")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Crash-safe, embeddable vector search engine")
set(CPACK_PACKAGE_DESCRIPTION "Vesper is a high-performance vector search engine designed for CPU-only environments, featuring crash-safe persistence, multiple index types, and SIMD acceleration.")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/vesper-arch/vesper")
set(CPACK_PACKAGE_CONTACT "vesper-dev@example.com")

# Version from VERSION file
if(EXISTS "${CMAKE_SOURCE_DIR}/VERSION")
    file(READ "${CMAKE_SOURCE_DIR}/VERSION" VESPER_VERSION)
    string(STRIP "${VESPER_VERSION}" VESPER_VERSION)
else()
    set(VESPER_VERSION "0.2.0")
endif()

string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ ${VESPER_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
set(CPACK_PACKAGE_VERSION ${VESPER_VERSION})

# License
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

# Installation directories
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "vesper")

# Components
set(CPACK_COMPONENTS_ALL libraries headers runtime documentation)
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Vesper shared and static libraries")
set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "Development Headers")
set(CPACK_COMPONENT_HEADERS_DESCRIPTION "C++ header files for Vesper development")
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Runtime")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Vesper runtime executables and tools")
set(CPACK_COMPONENT_DOCUMENTATION_DISPLAY_NAME "Documentation")
set(CPACK_COMPONENT_DOCUMENTATION_DESCRIPTION "Documentation and examples")

# Dependencies
set(CPACK_COMPONENT_HEADERS_DEPENDS libraries)
set(CPACK_COMPONENT_RUNTIME_DEPENDS libraries)

# Platform-specific settings
if(WIN32)
    # Windows (NSIS, ZIP)
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "Vesper ${VESPER_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "Vesper")
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_MODIFY_PATH ON)
    
    # Add to PATH
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
        WriteRegStr HKLM 'SYSTEM\\\\CurrentControlSet\\\\Control\\\\Session Manager\\\\Environment' 'VESPER_HOME' '$INSTDIR'
        SendMessage \${HWND_BROADCAST} \${WM_WININICHANGE} 0 'STR:Environment' /TIMEOUT=5000
    ")
    
    set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
        DeleteRegValue HKLM 'SYSTEM\\\\CurrentControlSet\\\\Control\\\\Session Manager\\\\Environment' 'VESPER_HOME'
        SendMessage \${HWND_BROADCAST} \${WM_WININICHANGE} 0 'STR:Environment' /TIMEOUT=5000
    ")
    
elseif(APPLE)
    # macOS (DragNDrop, TGZ)
    set(CPACK_GENERATOR "DragNDrop;TGZ")
    set(CPACK_DMG_VOLUME_NAME "Vesper ${VESPER_VERSION}")
    set(CPACK_DMG_FORMAT "UDZO")
    set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_SOURCE_DIR}/packaging/dmg-background.png")
    set(CPACK_DMG_DS_STORE_SETUP_SCRIPT "${CMAKE_SOURCE_DIR}/packaging/dmg-setup.scpt")
    
    # Bundle settings
    set(CPACK_BUNDLE_NAME "Vesper")
    set(CPACK_BUNDLE_PLIST "${CMAKE_SOURCE_DIR}/packaging/Info.plist")
    set(CPACK_BUNDLE_ICON "${CMAKE_SOURCE_DIR}/packaging/vesper.icns")
    
    # Code signing
    if(DEFINED ENV{APPLE_DEVELOPER_ID})
        set(CPACK_BUNDLE_APPLE_CERT_APP "$ENV{APPLE_DEVELOPER_ID}")
        set(CPACK_BUNDLE_APPLE_CODESIGN_PARAMETER "--deep --force --verify --verbose --options runtime")
    endif()
    
else()
    # Linux (DEB, RPM, TGZ)
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
    
    # DEB specific
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Vesper Maintainers")
    set(CPACK_DEBIAN_PACKAGE_SECTION "database")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11), libtbb2")
    
    # Architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
        set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
        set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
    endif()
    
    # Optional dependencies
    set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "libnuma1, liburing2")
    
    # RPM specific
    set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
    set(CPACK_RPM_PACKAGE_GROUP "Applications/Databases")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc >= 2.34, libstdc++ >= 11, tbb")
    set(CPACK_RPM_PACKAGE_SUGGESTS "numactl-libs, liburing")
    set(CPACK_RPM_PACKAGE_AUTOREQPROV ON)
    
    # Post-install scripts
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA 
        "${CMAKE_SOURCE_DIR}/packaging/debian/postinst"
        "${CMAKE_SOURCE_DIR}/packaging/debian/prerm")
    
    set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE 
        "${CMAKE_SOURCE_DIR}/packaging/rpm/postinst.sh")
    set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE 
        "${CMAKE_SOURCE_DIR}/packaging/rpm/prerm.sh")
endif()

# Source package
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "vesper-${VESPER_VERSION}-source")
set(CPACK_SOURCE_IGNORE_FILES
    "/\\\\.git/"
    "/\\\\.github/"
    "/build.*/"
    "/\\\\.vscode/"
    "/\\\\.idea/"
    "\\\\.DS_Store"
    ".*\\\\.swp"
    ".*\\\\.swo"
    ".*~"
)

# Include CPack
include(CPack)

# Component installation
include(CPackComponent)
cpack_add_component_group(Development
    DISPLAY_NAME "Development"
    DESCRIPTION "Headers and libraries for development"
)

cpack_add_component_group(Runtime
    DISPLAY_NAME "Runtime"
    DESCRIPTION "Runtime libraries and executables"
)

cpack_add_component(libraries
    DISPLAY_NAME "Libraries"
    DESCRIPTION "Vesper shared libraries"
    GROUP Runtime
    INSTALL_TYPES Full Minimal
)

cpack_add_component(headers
    DISPLAY_NAME "Headers"
    DESCRIPTION "Development header files"
    GROUP Development
    INSTALL_TYPES Full Developer
)

cpack_add_component(documentation
    DISPLAY_NAME "Documentation"
    DESCRIPTION "Documentation and examples"
    GROUP Development
    INSTALL_TYPES Full
)

# Installation types
cpack_add_install_type(Full DISPLAY_NAME "Full Installation")
cpack_add_install_type(Minimal DISPLAY_NAME "Minimal Installation")
cpack_add_install_type(Developer DISPLAY_NAME "Developer Installation")