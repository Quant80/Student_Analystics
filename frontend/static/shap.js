function renderShapChart(shapValues, chartId) {
    // Get the canvas element
    const ctx = document.getElementById(chartId).getContext('2d');

    // Extract labels (feature names) and SHAP values
    const labels = Object.keys(shapValues);
    const values = Object.values(shapValues);

    // Assign colors based on SHAP value sign (positive: blue, negative: red)
    const backgroundColors = values.map(value => 
        value >= 0 ? '#0d16c8' : '#ff0000'
    );

    // Create the horizontal bar chart
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Values',
                data: values,
                backgroundColor: backgroundColors,
                borderWidth: 0
            }]
        },
        options: {
            indexAxis: 'y', // Makes the bar chart horizontal
            responsive: true,
            plugins: {
                legend: {
                    display: false // Hide the default legend (we'll add a custom one in HTML)
                },
                title: {
                    display: true,
                    text: 'Feature Contributions (SHAP Values)',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: {
                        top: 10,
                        bottom: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.x;
                            return `SHAP Value: ${value.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'SHAP Value',
                        font: {
                            size: 14
                        }
                    },
                    grid: {
                        display: true
                    },
                    ticks: {
                        stepSize: 0.5
                    }
                },
                y: {
                    title: {
                        display: false
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}