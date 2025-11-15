idf_component_register(SRCS "test_app_main.c"
                            # Performance Tests (CORE)
                            "test_performance_suite.c"
                            "test_threshold_optimization.c"
                            "test_temporal_robustness.c"
                            "test_home_assistant_integration.c"
                            # Calibration & Optimization
                            "test_calibration.c"
                            "test_real_calibration.c"
                            "test_pca_subcarrier.c"
                            # Segmentation
                            "test_segmentation.c"
                            # Component Tests (Minimal)
                            "test_filters.c"
                            "test_wavelet.c"
                            # Main component sources
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/filters.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/wavelet.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/csi_processor.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/statistics.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/calibration.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/detection_engine.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/config_manager.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/nvs_storage.c"
                            "${CMAKE_CURRENT_LIST_DIR}/../../main/segmentation.c"
                       INCLUDE_DIRS "." "${CMAKE_CURRENT_LIST_DIR}/../../main"
                       REQUIRES unity esp_wifi nvs_flash esp_timer esp-dsp)
