import matplotlib.pyplot as plt
import serial
import numpy as np
from numpy.random import randn
from numpy import dot, linalg, eye, array
import matplotlib.animation as animation
from time import time
import sympy
from sympy.abc import alpha, x, y, v, w, R, theta, kappa, beta
from sympy import symbols, Matrix, Function, diff, preview
import scipy
import serial
import math
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints



class QuaterUpdate:
    def __init__(self, gain):
        self.gain = gain

    def update_quat(self, quat, gyro, acc, mag, dt):
        """
        Parameters
        ----------
        Update the quaternion using the gyro, accelerometer, and magnetometer data
        quat: Current quaternion
        gyro: Angular velocity in rad/s
        acc: Acceleration in m/s^2 (Normalized later on)
        mag: Magnetic field in uT (Normalized later on)
        dt: Time step in seconds
        """
        # Skip if no acceleration/gravity measured
        if np.linalg.norm(acc) == 0:
            return quat
        
        # update only using acc and gyro if no magnetometer reads
        if np.linalg.norm(mag) == 0:
            return self.update_quat_imu(quat, gyro, acc, dt)
        
        # find the quaternion update using gyro data
        # gyro data must be in rad/s
        gyro = np.array([0, *gyro])
        dq = 0.5 * self.quat_mult(quat, gyro)

        a_norm = np.linalg.norm(acc)
        if a_norm > 0:
            a = acc / a_norm
            m = mag / np.linalg.norm(mag)

            # Rotate normalized magnetometer measurements
            q_m = np.array([0, *m])
            h = self.quat_mult(
                quat,
                self.quat_mult(q_m, self.quat_conj(quat))
            )
            bx = np.linalg.norm([h[1], h[2]])
            bz = h[3]

            # Normalize quaternion
            qw, qx, qy, qz = (quat / np.linalg.norm(quat)).ravel()

            f = np.array([
                2.0*(qx*qz - qw*qy)   - a[0],
                2.0*(qw*qx + qy*qz)   - a[1],
                2.0*(0.5-qx**2-qy**2) - a[2],
                2.0*bx*(0.5 - qy**2 - qz**2) + 2.0*bz*(qx*qz - qw*qy)       - m[0],
                2.0*bx*(qx*qy - qw*qz)       + 2.0*bz*(qw*qx + qy*qz)       - m[1],
                2.0*bx*(qw*qy + qx*qz)       + 2.0*bz*(0.5 - qx**2 - qy**2) - m[2]
            ])

            if np.linalg.norm(f) > 0:
                J = np.array([[-2.0*qy,               2.0*qz,              -2.0*qw,               2.0*qx             ],
                              [ 2.0*qx,               2.0*qw,               2.0*qz,               2.0*qy             ],
                              [ 0.0,                 -4.0*qx,              -4.0*qy,               0.0                ],
                              [-2.0*bz*qy,            2.0*bz*qz,           -4.0*bx*qy-2.0*bz*qw, -4.0*bx*qz+2.0*bz*qx],
                              [-2.0*bx*qz+2.0*bz*qx,  2.0*bx*qy+2.0*bz*qw,  2.0*bx*qx+2.0*bz*qz, -2.0*bx*qw+2.0*bz*qy],
                              [ 2.0*bx*qy,            2.0*bx*qz-4.0*bz*qx,  2.0*bx*qw-4.0*bz*qy,  2.0*bx*qx          ]])

                # get gradient
                gradient = J.T @ f

                # get norm of gradient
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm > 0:
                    gradient /= gradient_norm
                dq -= self.gain * gradient
        
        quat_new = quat + dq * dt
        quat_new /= np.linalg.norm(quat_new)
        return quat_new

        
    def update_quat_imu(self, quat, gyro, acc, dt):
        if np.linalg.norm(gyro) == 0:
            return quat
        
        gyro = np.array([0, *gyro])
        dq = 0.5 * self.quat_mult(quat, gyro)
        
        a_norm = np.linalg.norm(acc)
        if a_norm > 0:
            a = acc / a_norm
            qw, qx, qy, qz = (quat / np.linalg.norm(quat)).ravel()
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2]])
            if np.linalg.norm(f) > 0:
                # Jacobian
                J = np.array([[-2.0*qy,  2.0*qz, -2.0*qw, 2.0*qx],
                              [ 2.0*qx,  2.0*qw,  2.0*qz, 2.0*qy],
                              [ 0.0,    -4.0*qx, -4.0*qy, 0.0   ]])
                
                gradient = J.T @ f
                gradient /= np.linalg.norm(gradient)
                dq -= self.gain * gradient
        
        quat_new = quat + dq * dt
        quat_new /= np.linalg.norm(quat_new)
        return quat_new


    def quat_mult(self, q, r):
        qw, qx, qy, qz = q.ravel()
        rw, rx, ry, rz = r.ravel()
        return np.array([
            qw * rw - qx * rx - qy * ry - qz * rz,
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw
        ])

    def quat_conj(self, q):
        qw, qx, qy, qz = q.ravel()
        return np.array([qw, -qx, -qy, -qz])
    
    def quat_to_rot(self, q):
        q /= np.linalg.norm(q)
        q0, q1, q2, q3 = q.ravel()
        return np.array([
            [1 - 2*(q2**2 + q3**2),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3),       1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)]
        ])
    
    def initial_quaternion(self, accel, mag):
        """
        Finds initial quaternion with respect to the global frame
        NED frame:
        X - Magnetic North
        Y - East
        Z - Gravity
        """
        # Normalize accelerometer measurement to get gravity direction.
        g = accel / np.linalg.norm(accel)
        
        # Normalize magnetometer measurement.
        m = mag / np.linalg.norm(mag)
        
        # Compute "east" vector (perpendicular to both m and g).
        east = np.cross(g, m)
        east /= np.linalg.norm(east)
        
        # Compute "north" vector (lies in the horizontal plane, perpendicular to east).
        north = np.cross(east, g)
        
        # Create the rotation matrix
        R = np.c_[north, east, g].T
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        # Return the quaternion [w, x, y, z]
        return np.array([w, x, y, z])


def run_orientation_tracker():
    # Set up the serial connection
    # port = "/dev/tty.usbmodem11301"
    # port = "/dev/cu.usbmodem11301"
    # port = "/dev/cu.usbmodem1201"
    port = "/dev/cu.usbmodem1301"
    # port = "/dev/cu.usbmodem11201"
    ser = serial.Serial(port, 115200)

    prev_time = 0
    prev = 0
    g = 9.81
    while True:
        # Read a line of data from the serial port
        line = ser.readline().decode('utf-8').strip().split(',')
        print(line, len(line))
        if not any([len(x) == 0 for x in line]) and len(line) == 11:
            prev_time = float(line[10])
            prev = float(line[9])
            break
    print("Processing started!")

    # Initial setup
    gain = 0.5
    quater = QuaterUpdate(gain)
    q0 = None
    x0 = np.array([1, 0, 0])
    y0 = np.array([0, 1, 0])
    z0 = np.array([0, 0, 1])

    # Create figure and axis ONCE outside the loop
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits to keep a consistent view
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    # Initialize the lines
    xs = [[0, 1], [0, 0], [0, 0]]
    ys = [[0, 0], [0, 1], [0, 0]]
    zs = [[0, 0], [0, 0], [0, 1]]

    # Create line objects ONCE
    x_line, = ax.plot(xs[0], xs[1], xs[2], 'r-', linewidth=2, label="X")
    y_line, = ax.plot(ys[0], ys[1], ys[2], 'g-', linewidth=2, label="Y")
    z_line, = ax.plot(zs[0], zs[1], zs[2], 'b-', linewidth=2, label="Z")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Add a title
    ax.set_title('IMU Orientation')

    # For smoother animation
    plt.ion()  # Turn on interactive mode
    fig.canvas.draw()
    plt.show(block=False)

    qs = []
    while True:
        # Read a line of data from the serial port
        line = ser.readline().decode('utf-8').strip().split(',')
        ax, ay, az, wx, wy, wz, mx, my, mz, dist, time = list(map(float, line))
        
        dt = (time - prev_time) / 1000
        prev_time = time

        # processing angular speed, convert to rad/s
        wx, wy, wz = (
            math.radians(wx), 
            math.radians(wy), 
            math.radians(wz)
        )

        if q0 is None:
            q0 = quater.initial_quaternion(
                accel=np.array([ax, ay, az]),
                mag=np.array([mx, my, mz])
            )
            continue
        # qs.append(q0)
        # if len(qs) >= 100:
        #     break

        q0 = quater.update_quat(
            quat=q0,
            gyro=np.array([wx, wy, wz]),
            acc=np.array([ax, ay, az]),
            mag=np.array([mx, my, mz]),
            dt=dt
        )
        print(q0)

        R = quater.quat_to_rot(q0)
        rx = R @ x0
        ry = R @ y0
        rz = R @ z0
        
        # Update the line data
        x_line.set_data([0, rx[0]], [0, rx[1]])
        x_line.set_3d_properties([0, rx[2]])
        
        y_line.set_data([0, ry[0]], [0, ry[1]])
        y_line.set_3d_properties([0, ry[2]])
        
        z_line.set_data([0, rz[0]], [0, rz[1]])
        z_line.set_3d_properties([0, rz[2]])
        
        # Redraw the figure without creating a new one
        fig.canvas.draw_idle()
        plt.pause(0.001)
        
        # Optional: Add a way to exit the loop
        try:
            plt.pause(0.001)
        except KeyboardInterrupt:
            break


    # print("Final avg: ", np.mean(qs, axis=0))
    

if __name__ == "__main__":
    run_orientation_tracker()
    