document.addEventListener('DOMContentLoaded', function () {
    console.log('script.js loaded');

    // Xử lý radio button feature option
    const featureDefaultRadio = document.getElementById('feature_default');
    const featureCustomRadio = document.getElementById('feature_custom');
    const featureCheckboxes = document.getElementById('featureCheckboxes');
    const featureCheckboxInputs = document.querySelectorAll('.feature-checkbox');
    const selectedCount = document.getElementById('selectedCount');

    // Only proceed with feature selection logic if elements exist
    if (featureDefaultRadio && featureCustomRadio && featureCheckboxes) {
        console.log('Found all feature selection elements');

        featureDefaultRadio.addEventListener('change', function () {
            console.log('Default radio selected');
            featureCheckboxes.classList.add('d-none');
            featureCheckboxInputs.forEach(function (checkbox) {
                checkbox.checked = false;
            });
            updateSelectedCount();
        });

        featureCustomRadio.addEventListener('change', function () {
            console.log('Custom radio selected');
            if (this.checked) {
                featureCheckboxes.classList.remove('d-none');
                console.log('Feature checkboxes should be visible');
            } else {
                featureCheckboxes.classList.add('d-none');
                console.log('Feature checkboxes should be hidden');
            }
        });        // Kiểm tra trạng thái ban đầu
        if (featureCustomRadio.checked) {
            featureCheckboxes.classList.remove('d-none');
            console.log('Initial state: Custom radio checked, showing checkboxes');
        }

        // Cập nhật số lượng feature đã chọn
        function updateSelectedCount() {
            if (selectedCount) {
                const count = featureDefaultRadio.checked ? 5 : document.querySelectorAll('.feature-checkbox:checked').length;
                selectedCount.textContent = 'Đã chọn: ' + count;
                console.log('Selected count updated:', count);
            }
        }

        // Sự kiện thay đổi checkbox
        if (featureCheckboxInputs.length > 0) {
            console.log('Found', featureCheckboxInputs.length, 'checkboxes');
            featureCheckboxInputs.forEach(function (checkbox) {
                checkbox.addEventListener('change', function () {
                    updateSelectedCount();
                    console.log('Checkbox changed:', checkbox.value, checkbox.checked);
                });
            });
        } else {
            console.warn('No feature checkboxes found');
        }

        // Initial count update
        updateSelectedCount();

        // Reset features
        const resetFeaturesBtn = document.getElementById('resetFeatures');
        if (resetFeaturesBtn) {
            resetFeaturesBtn.addEventListener('click', function () {
                console.log('Reset features clicked');
                featureDefaultRadio.checked = true;
                featureCustomRadio.checked = false;
                featureCheckboxes.classList.add('d-none');
                featureCheckboxInputs.forEach(function (checkbox) {
                    checkbox.checked = false;
                });
                updateSelectedCount();
            });
        }
    } else {
        console.log('Feature selection elements not found on this page - skipping feature selection logic');
    }    // Xử lý radio button PCA/không PCA
    const processMethodRadios = document.querySelectorAll('input[name="process_method"]');
    const pcaOptions = document.querySelector('.pca-options');
    if (processMethodRadios.length > 0 && pcaOptions) {
        processMethodRadios.forEach(function (radio) {
            radio.addEventListener('change', function () {
                pcaOptions.classList.toggle('d-none', this.value !== 'pca');
                const explainedVariance = document.getElementById('explained_variance');
                if (explainedVariance) {
                    if (this.value !== 'pca') {
                        explainedVariance.removeAttribute('required');
                    } else {
                        explainedVariance.setAttribute('required', 'required');
                    }
                }
            });
        });
    }    // Progress bar cho form
    const forms = [
        document.getElementById('uploadForm'),
        document.getElementById('featureForm'),
        document.getElementById('analysisForm'),
        document.getElementById('modelForm')
    ];
    forms.forEach(function (form) {
        if (form) {
            form.addEventListener('submit', function () {
                const progress = form.nextElementSibling;
                if (progress && progress.classList.contains('progress')) {
                    progress.classList.remove('d-none');
                }
            });
        }
    });

    // Modal plot zoom
    const plotModal = document.getElementById('plotModal');
    if (plotModal) {
        plotModal.addEventListener('show.bs.modal', function (event) {
            const img = event.relatedTarget.getAttribute('data-img');
            const modalImg = document.getElementById('modalPlotImg');
            if (modalImg) {
                modalImg.src = img;
            }
        });
    }
});