var map, heatmap;

// Heatmap data: 500 Points

function initMap() {
    var bounds = new google.maps.LatLngBounds();

    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 3,
        center: {
            lat: 0,
            lng: 0
        },
        mapTypeId: google.maps.MapTypeId.ROADMAP
    });

    function fitForBounds(data) {
        for (var i = 0; i < data.length; i++) {
            bounds.extend(data[i]);
        }
    }
    fitForBounds(getPoints());
    map.setCenter(bounds.getCenter());
    heatmap = new google.maps.visualization.HeatmapLayer({
        data: getPoints(),
        map: map
    });
    map.fitBounds(bounds);
}

function toggleHeatmap() {
    heatmap.setMap(heatmap.getMap() ? null : map);
}

function changeGradient() {
    var gradient = [
        'rgba(0, 255, 255, 0)',
        'rgba(0, 255, 255, 1)',
        'rgba(0, 191, 255, 1)',
        'rgba(0, 127, 255, 1)',
        'rgba(0, 63, 255, 1)',
        'rgba(0, 0, 255, 1)',
        'rgba(0, 0, 223, 1)',
        'rgba(0, 0, 191, 1)',
        'rgba(0, 0, 159, 1)',
        'rgba(0, 0, 127, 1)',
        'rgba(63, 0, 91, 1)',
        'rgba(127, 0, 63, 1)',
        'rgba(191, 0, 31, 1)',
        'rgba(255, 0, 0, 1)'
    ]
    heatmap.set('gradient', heatmap.get('gradient') ? null : gradient);
}

function changeRadius() {
    heatmap.set('radius', heatmap.get('radius') ? null : 20);
}

function changeOpacity() {
    heatmap.set('opacity', heatmap.get('opacity') ? null : 0.2);
}

function getPoints() {
    return [new google.maps.LatLng(1.2784698, 103.839155), new google.maps.LatLng(1.3584781, 103.972429793384), new google.maps.LatLng(1.36082365, 103.972004451282), new google.maps.LatLng(1.349272, 103.706747), new google.maps.LatLng(1.44900405, 103.782034756083), new google.maps.LatLng(1.3658773, 103.977475414943), new google.maps.LatLng(1.350269, 103.708474), new google.maps.LatLng(1.4523215, 103.781881335281), new google.maps.LatLng(1.3417087, 103.94034047556), new google.maps.LatLng(1.3235219, 103.844209183162), new google.maps.LatLng(1.311915, 103.84650310388), new google.maps.LatLng(1.369869, 103.851713520552), new google.maps.LatLng(1.33984035, 103.850172549507), new google.maps.LatLng(1.3242583, 103.845464326322), new google.maps.LatLng(1.2788135, 103.8396105), new google.maps.LatLng(1.29898555, 103.83984104377), new google.maps.LatLng(1.2784698, 103.839155), new google.maps.LatLng(1.3059053, 103.822256131264), new google.maps.LatLng(1.2952466, 103.800682144999)];
}