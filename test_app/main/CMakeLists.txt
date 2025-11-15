idf_component_register(SRCS "test_app_main.c"
                            # Performance Tests (CORE) - Segmentation-based
                            "test_performance_suite.c"          # PRIMARY: Segmentation metrics
                            "test_threshold_optimization.c"     # SECONDARY: Feature ranking (features_enabled)
                            # Segmentation Unit Tests
                            "test_segmentation.c"
                            # Tuning & Feature Tests (SECONDARY)
                            "test_segmentation_tuning.c"
                            "test_features.c"
                            "test_pca_subcarrier.c"
                            # Component Tests (Minimal)
                            "test_filters.c"
                            "test_wavelet.c"
                            # Main component sources
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/filters.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/wavelet.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/csi_processor.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/config_manager.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/nvs_storage.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/segmentation.c"
                       INCLUDE_DIRS "." "${CMAKE_CURRENT_LIST_DIR}/../../main"
                       REQUIRES unity esp_wifi nvs_flash esp_timer esp-dsp)
