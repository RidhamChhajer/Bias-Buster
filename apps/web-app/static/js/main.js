// Main JavaScript for Bias Buster Platform

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Form validation enhancements
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        if (alert.classList.contains('alert-success')) {
            setTimeout(() => {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }, 5000);
        }
    });

    // Add animation classes on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    });

    // Observe cards and sections
    document.querySelectorAll('.card, .hero-section').forEach(el => {
        observer.observe(el);
    });

    // Copy to clipboard functionality
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(function() {
            showNotification('Copied to clipboard!', 'success');
        });
    }

    // Show notifications
    function showNotification(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alertDiv);
            bsAlert.close();
        }, 3000);
    }

    // Global error handler
    window.addEventListener('error', function(e) {
        console.error('JavaScript Error:', e.error);
        showNotification('An error occurred. Please refresh the page.', 'danger');
    });
});

// Utility functions
const BiasUtils = {
    formatPercentage: function(value, decimals = 1) {
        return (value * 100).toFixed(decimals) + '%';
    },
    
    formatNumber: function(value, decimals = 2) {
        return parseFloat(value).toFixed(decimals);
    },
    
    getRiskColor: function(riskLevel) {
        const colors = {
            'CRITICAL': '#dc3545',
            'HIGH': '#fd7e14', 
            'MEDIUM': '#ffc107',
            'LOW': '#17a2b8',
            'MINIMAL': '#28a745'
        };
        return colors[riskLevel] || '#6c757d';
    },
    
    downloadData: function(data, filename, type = 'application/json') {
        const blob = new Blob([data], { type: type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
};

// Export for use in other scripts
window.BiasUtils = BiasUtils;
