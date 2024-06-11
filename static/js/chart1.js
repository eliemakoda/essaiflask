const ctx = document.getElementById('barchart').getContext('2d');
const barchart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Sampaloc', 'Quiling', 'San Isidro', 'Buso-Buso', 'Leviste', 'Bañaga', 'Manalao', 'Mataas na Kahoy', 'Tanauan', 'Pulang Bato'],
        datasets: [{
            label: 'pH',
            data: [8.05, 8.06, 8.01, 8.04, 8.02, 8, 8.19, , 8.15, 8.11],
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
        },
        {
            label: 'Ammonia',
            data: [0.66, 0.56, 0.56, 0.61, 0.56, 0.63, 0.5, , 0.53, 0.39],
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
        },
        {
            label: 'DO',
            data: [4.95, 3.37, 3.6, 3.04, 4.41, 2.34, 5.27, , 4.1, 4.74],
            backgroundColor: 'rgba(255, 206, 86, 0.2)',
            borderColor: 'rgba(255, 206, 86, 1)',
            borderWidth: 1
        },
        {
            label: 'Nitrate',
            data: [0.06, 0.14, 0.14, 0.08, 0.11, 0.19, 0.08, , 0.12, 0.14],
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        },
        {
            label: 'Phosphate',
            data: [2.39, 2.27, 2.38, 2.31, 2.27, 2.27, 2.22, , 2.24, 2.26],
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            borderColor: 'rgba(153, 102, 255, 1)',
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginatZero: true
            }
        }
    }
});


