from djitellopy import Tello
import time

# Initialize the drone object
tello = Tello()

# Establish the connection
print("Connecting to Tello...")
tello.connect()

# Query battery status
battery_start = tello.get_battery()
print(f"Connection Successful! Battery: {battery_start}%")

try:
    tello.takeoff()
    print("Takeoff successful! Hovering for 3 seconds...")
    time.sleep(10)  # Give the drone time to stabilize
    
    battery_hover = tello.get_battery()
    print("Battery after 10s hover:", battery_hover, "%")

    print("Attempting to land...")
    tello.land()
    print("Landing successful!")
    

    battery_end = tello.get_battery()
    print("Battery after landing:", battery_end, "%")

    drop = battery_start - battery_hover
    print("Battery drop during hover:", drop, "%")

except Exception as e:
    print(f"Error occurred: {e}")
    print("Attempting emergency landing...")
    try:
        # Try to send land command without retry limit
        tello.send_command_without_return("land")
        time.sleep(3)
    except:
        print("Emergency landing command sent. Please manually catch the drone if needed!")


# Disconnect
tello.end()