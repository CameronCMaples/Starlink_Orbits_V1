import requests
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M_EARTH = 5.972e24  # Mass of the Earth (kg)
R_EARTH = 6371e3  # Radius of the Earth (m)

# Create Satellite Class
class Satellite:
    def __init__(self, name, object_id, epoch, mean_motion, eccentricity, inclination, \
                 ra_of_asc_node, arg_of_pericenter, mean_anomaly, classification_type, norad_cat_id, \
                    element_set_no, rev_at_epoch, bstar, mean_motion_dot, mean_motion_ddot):
        
        self.name = name
        self.object_id = object_id
        self.epoch = epoch
        self.mean_motion = mean_motion
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.ra_of_asc_node = ra_of_asc_node
        self.arg_of_pericenter = arg_of_pericenter
        self.mean_anomaly = mean_anomaly
        self.classification_type = classification_type
        self.norad_cat_id = norad_cat_id
        self.element_set_no = element_set_no
        self.rev_at_epoch = rev_at_epoch
        self.bstar = bstar
        self.mean_motion_dot = mean_motion_dot
        self.mean_motion_ddot = mean_motion_ddot

# URL for the Starlink satellite data in JSON format
url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json"

def fetch_satellites(url):
    response = requests.get(url)
    satellites_data = response.json()

    satellites = []

    for data in satellites_data:
        satellite = Satellite(
            name=data.get("OBJECT_NAME"),
            object_id=data.get("OBJECT_ID"),
            epoch=data.get("EPOCH"),
            mean_motion=data.get("MEAN_MOTION"),
            eccentricity=data.get("ECCENTRICITY"),
            inclination=data.get("INCLINATION"),
            ra_of_asc_node=data.get("RA_OF_ASC_NODE"),
            arg_of_pericenter=data.get("ARG_OF_PERICENTER"),
            mean_anomaly=data.get("MEAN_ANOMALY"),
            classification_type=data.get("CLASSIFICATION_TYPE"),
            norad_cat_id=data.get("NORAD_CAT_ID"),
            element_set_no=data.get("ELEMENT_SET_NO"),
            rev_at_epoch=data.get("REV_AT_EPOCH"),
            bstar=data.get("BSTAR"),
            mean_motion_dot=data.get("MEAN_MOTION_DOT"),
            mean_motion_ddot=data.get("MEAN_MOTION_DDOT")
        )
        satellites.append(satellite)

    return satellites

SatelliteData = fetch_satellites(url)
total_satellites = len(SatelliteData)

# Number of Starlink Satellites in orbit
print("There are currently: " + str(len(SatelliteData)) + " Starlink Satellites in orbit.")

def plot_satellite_orbit(satellite_name):
    # Fetch all Starlink satellite data 
    SatelliteData = fetch_satellites(url)
    
    #Fetch data of specified Starlink satellite
    selected_satellite = next((sat for sat in SatelliteData if sat.name == satellite_name), None)
    
    if selected_satellite is None:
        print(f"Satellite named {satellite_name} not found.")
        return
        

    # Solve Kepler's equation
    def kepler_equation(e, M, tol=1e-10, max_iter=100):
        """
        Solve Kepler's equation for the eccentric anomaly using Newton's method.

        Parameters:
        e (float): Eccentricity of the orbit
        M (float): Mean anomaly in radians
        tol (float, optional): Tolerance for convergence. Default is 1e-10.
        max_iter (int, optional): Maximum number of iterations. Default is 100.

        Returns:
        float: Eccentric anomaly in radians
        """
        E = M  # Initial guess for eccentric anomaly
        for i in range(max_iter):
            f = E - e * np.sin(E) - M
            df = 1 - e * np.cos(E)
            E_new = E - f / df
            if np.abs(E_new - E) < tol:
                return E_new
            E = E_new
        else:
            raise RuntimeError("Kepler's equation did not converge.")

    # Convert units and calculate orbital elements
    epoch = "2024-04-10T03:49:45.167520"
    eccentricity = selected_satellite.eccentricity
    inclination = selected_satellite.inclination * np.pi / 180  # Convert to radians
    ra_of_asc_node = selected_satellite.ra_of_asc_node * np.pi / 180  # Convert to radians
    arg_of_pericenter = selected_satellite.arg_of_pericenter * np.pi / 180  # Convert to radians
    mean_anomaly = selected_satellite.mean_anomaly * np.pi / 180  # Convert to radians
    mean_motion = selected_satellite.mean_motion * 2 * np.pi / 86400  # Convert to rad/s

    # Calculate the semi-major axis from mean motion
    GM_EARTH = G * M_EARTH
    period = 2 * np.pi / mean_motion
    semi_major_axis = (GM_EARTH * period ** 2 / (4 * np.pi ** 2)) ** (1 / 3)

    # Calculate the orbital elements
    a = semi_major_axis
    e = eccentricity
    i = inclination
    Omega = ra_of_asc_node
    omega = arg_of_pericenter
    M = mean_anomaly

    # Calculate the position and velocity vectors
    E = kepler_equation(e, M)  # Solve Kepler's equation for the eccentric anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))  # True anomaly

    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))  # Radius
    x = r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i))
    y = r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i))
    z = r * np.sin(omega + nu) * np.sin(i)

    vr = np.sqrt(GM_EARTH / a) * e * np.sin(nu) / (1 + e * np.cos(nu))  # Radial velocity
    vt = np.sqrt(GM_EARTH / a) * (1 + e * np.cos(nu))  # Transverse velocity
    vx = vr * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i)) + \
        vt * (-np.sin(Omega) * np.sin(omega + nu) - np.cos(Omega) * np.cos(omega + nu) * np.cos(i))
    vy = vr * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i)) + \
        vt * (np.cos(Omega) * np.sin(omega + nu) - np.sin(Omega) * np.cos(omega + nu) * np.cos(i))
    vz = vr * np.sin(omega + nu) * np.sin(i) + vt * (np.cos(Omega) * np.cos(omega + nu) * np.sin(i) + np.sin(Omega) * np.sin(omega + nu) * np.sin(i))

    # Number of points to plot along the orbit
    num_points = 100

    # True anomaly range from 0 to 2*pi radians
    true_anomalies = np.linspace(0, 2 * np.pi, num_points)

    # Lists to store orbit points
    orbit_x, orbit_y, orbit_z = [], [], []

    for nu in true_anomalies:
        r = a * (1 - e ** 2) / (1 + e * np.cos(nu))  # Radius
        x = r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i))
        y = r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i))
        z = r * np.sin(omega + nu) * np.sin(i)
        
        orbit_x.append(x)
        orbit_y.append(y)
        orbit_z.append(z)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Earth
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth = R_EARTH * np.cos(u) * np.sin(v)
    y_earth = R_EARTH * np.sin(u) * np.sin(v)
    z_earth = R_EARTH * np.cos(v)
    ax.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.5)

    # Plot the orbit
    ax.plot(orbit_x, orbit_y, orbit_z, 'r-')

    ax.set_xlabel("X (In thousands of km)")
    ax.set_ylabel("Y (In thousands of km)")
    ax.set_zlabel("Z (In thousands of km)")
    ax.set_title("Orbit of " + selected_satellite.name)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Display
    attributes_text = f"""
    Satellite Name: {selected_satellite.name}
    Inclination: {selected_satellite.inclination} degrees
    Mean Anomoly: {selected_satellite.mean_anomaly}
    RA of Ascending Node: {selected_satellite.ra_of_asc_node} degrees
    Eccentricity: {selected_satellite.eccentricity}
    """

    plt.figtext(0.02, 0.5, attributes_text, ha="left", fontsize=10, \
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})


    plt.show()

initial_satellite_name = input("Enter the name of the Starlink satellite: ").upper()
plot_satellite_orbit(initial_satellite_name)

# Loop to allow replotting based on new Starlink choice
while True:
    next_satellite_name = input("Enter another Starlink satellite name to plot (or type 'exit' to quit): ").upper()
    if next_satellite_name.lower() == 'exit':
        break
    plot_satellite_orbit(next_satellite_name)