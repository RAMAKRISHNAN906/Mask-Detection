/* ═══════════════════════════════════════════════════════════════
   MaskGuard AI - Dashboard Charts Module
   ═══════════════════════════════════════════════════════════════ */

function initDashboardCharts(stats) {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#a0a0c0' : '#4a4a6a';
    const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';

    Chart.defaults.color = textColor;
    Chart.defaults.font.family = "'Inter', sans-serif";

    // ── Pie Chart ─────────────────────────────────────────────
    const pieCtx = document.getElementById('pieChart');
    if (pieCtx) {
        new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: ['With Mask', 'Without Mask'],
                datasets: [{
                    data: [stats.total_masks || 0, stats.total_no_masks || 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(244, 63, 94, 0.8)'
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(244, 63, 94, 1)'
                    ],
                    borderWidth: 2,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            pointStyleWidth: 12,
                            font: { size: 13, weight: '500' }
                        }
                    },
                    tooltip: {
                        backgroundColor: isDark ? 'rgba(15,15,40,0.9)' : 'rgba(255,255,255,0.95)',
                        titleColor: isDark ? '#f0f0ff' : '#1a1a2e',
                        bodyColor: isDark ? '#a0a0c0' : '#4a4a6a',
                        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8
                    }
                }
            }
        });
    }

    // ── Line Chart ────────────────────────────────────────────
    const lineCtx = document.getElementById('lineChart');
    if (lineCtx && stats.daily) {
        const labels = stats.daily.map(d => d.date);
        const maskData = stats.daily.map(d => d.masks || 0);
        const noMaskData = stats.daily.map(d => d.no_masks || 0);

        new Chart(lineCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'With Mask',
                        data: maskData,
                        borderColor: 'rgba(16, 185, 129, 1)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        borderWidth: 2
                    },
                    {
                        label: 'No Mask',
                        data: noMaskData,
                        borderColor: 'rgba(244, 63, 94, 1)',
                        backgroundColor: 'rgba(244, 63, 94, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: gridColor },
                        ticks: { font: { size: 11 } }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: {
                            font: { size: 11 },
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            font: { size: 13, weight: '500' }
                        }
                    },
                    tooltip: {
                        backgroundColor: isDark ? 'rgba(15,15,40,0.9)' : 'rgba(255,255,255,0.95)',
                        titleColor: isDark ? '#f0f0ff' : '#1a1a2e',
                        bodyColor: isDark ? '#a0a0c0' : '#4a4a6a',
                        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8
                    }
                }
            }
        });
    }

    // ── Bar Chart (Sources) ───────────────────────────────────
    const barCtx = document.getElementById('barChart');
    if (barCtx && stats.sources) {
        const sourceLabels = Object.keys(stats.sources).map(s =>
            s.charAt(0).toUpperCase() + s.slice(1)
        );
        const sourceData = Object.values(stats.sources);
        const sourceColors = {
            'Webcam': 'rgba(0, 240, 255, 0.7)',
            'Image': 'rgba(168, 85, 247, 0.7)',
            'Video': 'rgba(244, 63, 94, 0.7)'
        };

        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: sourceLabels,
                datasets: [{
                    label: 'Detections',
                    data: sourceData,
                    backgroundColor: sourceLabels.map(l => sourceColors[l] || 'rgba(0,240,255,0.5)'),
                    borderColor: sourceLabels.map(l => (sourceColors[l] || '').replace('0.7', '1')),
                    borderWidth: 2,
                    borderRadius: 8,
                    barPercentage: 0.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 13, weight: '600' } }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: {
                            font: { size: 11 },
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: isDark ? 'rgba(15,15,40,0.9)' : 'rgba(255,255,255,0.95)',
                        titleColor: isDark ? '#f0f0ff' : '#1a1a2e',
                        bodyColor: isDark ? '#a0a0c0' : '#4a4a6a',
                        borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8
                    }
                }
            }
        });
    }
}
