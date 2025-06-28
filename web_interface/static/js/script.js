document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    const vehicleStatusSpan = document.getElementById('vehicle-status');
    const targetSpeedSpan = document.getElementById('target-speed');
    const steeringAngleSpan = document.getElementById('steering-angle');
    const obstacleDetectedSpan = document.getElementById('obstacle-detected');
    const obstacleDistanceSpan = document.getElementById('obstacle-distance');
    const trafficSignsSpan = document.getElementById('traffic-signs');
    const laneInfoSpan = document.getElementById('lane-info');
    const systemStatusSpan = document.getElementById('system-status');
    const arduinoStatusSpan = document.getElementById('arduino-status');
    const zedCameraStatusSpan = document.getElementById('zed-camera-status');

    // Tab switching logic
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;

            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Hata ve sistem durumu için toast/alert fonksiyonu
    function showToast(message, type = 'info') {
        let toast = document.createElement('div');
        toast.className = `custom-toast custom-toast-${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => { toast.classList.add('show'); }, 100);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 500);
        }, 3500);
    }

    // Function to fetch and update status data
    async function updateStatus() {
        try {
            const response = await fetch('/status_feed');
            const data = await response.json();
            if (data.error) {
                showToast('Sistem hatası: ' + data.error, 'error');
            }

            // Update vehicle control data
            if (data.direction_data) {
                vehicleStatusSpan.textContent = data.direction_data.vehicle_status || 'N/A';
                targetSpeedSpan.textContent = data.direction_data.target_speed !== undefined ? data.direction_data.target_speed : 'N/A';
                steeringAngleSpan.textContent = data.direction_data.steering_angle !== undefined ? data.direction_data.steering_angle : 'N/A';
            } else {
                vehicleStatusSpan.textContent = 'No Data';
                targetSpeedSpan.textContent = 'No Data';
                steeringAngleSpan.textContent = 'No Data';
            }

            // Update obstacle data
            if (data.obstacle_results) {
                obstacleDetectedSpan.textContent = data.obstacle_results.obstacle_detected ? 'Yes' : 'No';
                obstacleDistanceSpan.textContent = data.obstacle_results.distance !== undefined ? `${data.obstacle_results.distance.toFixed(2)} mm` : 'N/A';
                if (data.obstacle_results.status && data.obstacle_results.status.includes("not possible")) {
                    obstacleDetectedSpan.textContent = 'N/A';
                    obstacleDistanceSpan.textContent = data.obstacle_results.status;
                }
            } else {
                obstacleDetectedSpan.textContent = 'No Data';
                obstacleDistanceSpan.textContent = 'No Data';
            }


            // Update traffic signs
            if (data.detection_results && data.detection_results.traffic_signs && data.detection_results.traffic_signs.length > 0) {
                trafficSignsSpan.textContent = data.detection_results.traffic_signs.map(sign => sign.label).join(', ');
            } else {
                trafficSignsSpan.textContent = 'None detected';
            }

            // Update lane info
            if (data.lane_results && data.lane_results.lanes && data.lane_results.lanes.length > 0) {
                laneInfoSpan.textContent = `${data.lane_results.lanes.length} lanes detected`;
            } else {
                laneInfoSpan.textContent = 'No lanes detected';
            }

            // Update system status
            systemStatusSpan.textContent = 'Operational'; // Or more detailed status based on sub-components
            arduinoStatusSpan.textContent = data.arduino_status || 'Unknown';
            zedCameraStatusSpan.textContent = data.zed_camera_status || 'Unknown';

        } catch (error) {
            console.error('Error fetching status data:', error);
            showToast('Sunucuya erişilemiyor!', 'error');
            systemStatusSpan.textContent = 'Error';
            arduinoStatusSpan.textContent = 'Error';
            zedCameraStatusSpan.textContent = 'Error';
        }
    }

    // Fetch status every 500ms
    setInterval(updateStatus, 500);

    // Initial status update
    updateStatus();
});

// Material UI + React ile modern arayüz
const { Button, AppBar, Toolbar, Typography, Container, Paper, Grid, Card, CardContent, CircularProgress, Box } = MaterialUI;

function StatusCard({ title, value }) {
    return (
        React.createElement(Card, { sx: { minWidth: 200, m: 1 } },
            React.createElement(CardContent, null,
                React.createElement(Typography, { variant: "h6", color: "text.secondary" }, title),
                React.createElement(Typography, { variant: "h5" }, value)
            )
        )
    );
}

function App() {
    const [status, setStatus] = React.useState(null);
    React.useEffect(() => {
        const interval = setInterval(() => {
            fetch('/status_feed').then(r => r.json()).then(setStatus);
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    return (
        React.createElement(Box, { sx: { flexGrow: 1 } },
            React.createElement(AppBar, { position: "static" },
                React.createElement(Toolbar, null,
                    React.createElement(Typography, { variant: "h6", sx: { flexGrow: 1 } }, "Dursun Kontrol Paneli"),
                    React.createElement(Button, { color: "inherit", href: "#" }, "Yenile")
                )
            ),
            React.createElement(Container, { sx: { mt: 4 } },
                React.createElement(Grid, { container: true, spacing: 2 },
                    React.createElement(Grid, { item: true, xs: 12, md: 8 },
                        React.createElement(Paper, { elevation: 3, sx: { p: 2, mb: 2 } },
                            React.createElement(Typography, { variant: "h6", sx: { mb: 2 } }, "Kamera Akışı"),
                            React.createElement("div", { style: { textAlign: 'center' } },
                                React.createElement("img", {
                                    src: "/video_feed",
                                    alt: "Kamera Akışı",
                                    style: { maxWidth: '100%', borderRadius: 8, background: '#222' }
                                })
                            )
                        )
                    ),
                    React.createElement(Grid, { item: true, xs: 12, md: 4 },
                        React.createElement(Paper, { elevation: 3, sx: { p: 2 } },
                            React.createElement(Typography, { variant: "h6", sx: { mb: 2 } }, "Durumlar"),
                            status ? (
                                React.createElement(Box, null,
                                    React.createElement(StatusCard, { title: "ZED Kamera", value: status.zed_camera_status }),
                                    React.createElement(StatusCard, { title: "Arduino", value: status.arduino_status }),
                                    React.createElement(StatusCard, { title: "Yol Algılama", value: JSON.stringify(status.lane_results) }),
                                    React.createElement(StatusCard, { title: "Nesne Tespiti", value: JSON.stringify(status.detection_results) }),
                                    React.createElement(StatusCard, { title: "Engel Analizi", value: JSON.stringify(status.obstacle_results) }),
                                    React.createElement(StatusCard, { title: "Yönlendirme", value: JSON.stringify(status.direction_data) })
                                )
                            ) : (
                                React.createElement(Box, { sx: { display: 'flex', justifyContent: 'center', mt: 4 } },
                                    React.createElement(CircularProgress, null)
                                )
                            )
                        )
                    )
                )
            )
        )
    );
}

ReactDOM.render(React.createElement(App), document.getElementById('root'));

// Toast için basit CSS ekle (otomatik eklenir)
(function addToastStyles() {
    const style = document.createElement('style');
    style.innerHTML = `
    .custom-toast {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        background: #333;
        color: #fff;
        padding: 14px 32px;
        border-radius: 8px;
        font-size: 1.1em;
        opacity: 0;
        pointer-events: none;
        z-index: 9999;
        transition: opacity 0.4s, bottom 0.4s;
    }
    .custom-toast.show {
        opacity: 1;
        bottom: 60px;
        pointer-events: auto;
    }
    .custom-toast-error { background: #c62828; }
    .custom-toast-info { background: #1976d2; }
    .custom-toast-success { background: #388e3c; }
    `;
    document.head.appendChild(style);
})();
