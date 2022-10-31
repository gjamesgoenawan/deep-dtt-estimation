"""
PI_custom_plugin

Provides tool to place a static camera on screen, record game screens, and plane details.

"""

from multiprocessing import current_process
import xp
import os
import time
import pyautogui
import pickle as pkl
import threading
import asyncio
import geopy.distance
import statistics


try:
    from XPListBox import XPCreateListBox
except ImportError:
    print("XPListBox is a custom python file provided with XPPython3, and required by this example you could copy it into PythonPlugins folder")
    raise

class PythonInterface:
    def __init__(self):
        self.Name = "Simulation Data Fetcher v1.1"
        self.Sig = "camera1.demos.xppython3"
        self.Desc = "An example using the camera module."
        self.plane_lat = None
        self.plane_lon = None
        self.plane_el = None
        self.HotKey_1 = None
        self.HotKey_2 = None
        self.HotKey_3 = None
        self.WindowId = None
        self.current_lat = None
        self.current_lon = None
        self.current_el = None
        self.currently_recording = False
        self.camera_static_lat = [1.355, 1.359]
        self.camera_static_lon = [103.984, 103.989]
        self.camera_static_alt = [24.927, 97.021]
        self.camera_static_pitch = [2.501, 1.916]
        self.camera_static_heading = [207.332, 209.162]
        self.camera_static_roll = [0, 0]
        self.camera_static_zoom = [10,8]
        a = xp.worldToLocal(self.camera_static_lat[0], self.camera_static_lon[0], self.camera_static_alt[0])
        self.a = a
        b = xp.worldToLocal(self.camera_static_lat[1], self.camera_static_lon[1], self.camera_static_alt[1])
        self.camera_static_local_x = [a[0], b[0]]
        self.camera_static_local_y = [a[1], b[1]]
        self.camera_static_local_z = [a[2], b[2]]
        self.DataMovingLoopID = xp.createFlightLoop(self.MoveFileCallback)
        self.selected_camera = 0
        self.recording_sequence_status = None

    def XPluginStart(self):
        # Prefetch the sim variables we will use.
        self.plane_lat = xp.findDataRef("sim/flightmodel/position/latitude")
        self.plane_lon = xp.findDataRef("sim/flightmodel/position/longitude")
        self.plane_el = xp.findDataRef("sim/flightmodel/position/elevation")
        self.camera_x = xp.findDataRef("sim/graphics/view/view_x")
        self.camera_y = xp.findDataRef("sim/graphics/view/view_y")
        self.camera_z = xp.findDataRef("sim/graphics/view/view_z")
        self.camera_pitch = xp.findDataRef("sim/graphics/view/view_pitch")
        self.camera_roll = xp.findDataRef("sim/graphics/view/view_roll")
        self.camera_heading = xp.findDataRef("sim/graphics/view/view_heading")
        #self.HotKey_1 = xp.registerHotKey(xp.VK_F9, xp.DownFlag, "Save Video and PKL of last recording", self.SaveDataCallback, 0)
        self.HotKey_0 = xp.registerHotKey(xp.VK_F8, xp.DownFlag, "Trigger Static External View 2", self.CameraHotKeyCallback_2, 0)
        self.HotKey_1 = xp.registerHotKey(xp.VK_F7, xp.DownFlag, "Trigger Static External View 1", self.CameraHotKeyCallback_1, 0)
        self.HotKey_2 = xp.registerHotKey(xp.VK_F6, xp.DownFlag, "Toggle Logging Action", self.LoggingHotKeyCallback, 0)
        self.HotKey_3 = xp.registerHotKey(xp.VK_F4, xp.DownFlag, "Toggle Monitoring Window", self.MonitoringHotkeyCallback, 0)
        self.windowInfo = (50, 600, 300, 400, 1,
                      self.DrawWindowCallback,
                      self.MouseClickCallback,
                      self.KeyCallback,
                      self.CursorCallback,
                      self.MouseWheelCallback,
                      0,
                      xp.WindowDecorationRoundRectangle,
                      xp.WindowLayerFloatingWindows,
                      None)
        for i in os.listdir("C:\\X-Plane 11\\Output"):
            if '.avi' in i:
                os.remove(f'C:\\X-Plane 11\\Output\\{i}')

        xp.registerFlightLoopCallback(self.RecordingSequenceLoop, -1, 0)

        return self.Name, self.Sig, self.Desc

    def XPluginStop(self):
        xp.unregisterHotKey(self.HotKey_1)
        xp.unregisterHotKey(self.HotKey_0)
        xp.unregisterHotKey(self.HotKey_2)
        xp.unregisterHotKey(self.HotKey_3)
        if self.WindowId != None:
            xp.destroyWindow(self.WindowId)
        xp.destroyFlightLoop(self.DataMovingLoopID)

    def XPluginEnable(self):
        return 1

    def XPluginDisable(self):
        pass    

    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        pass
    
    def RecordingSequenceLoop(self, elapsedMe, elapsedSim, counter, refcon):
        if self.recording_sequence_status == None:
            if self.currently_recording == True and self.selected_camera == 1   :
                self.recording_sequence_status = 0
            return -1
        elif self.recording_sequence_status == 0:
            if self.currently_recording == False:
                self.recording_sequence_status = 1
                pyautogui.hotkey('F5')
            return 1
        elif self.recording_sequence_status == 1:
            if self.selected_camera == 1:
                pyautogui.hotkey('F8')
                self.recording_sequence_status = 2
            return 1
        elif self.recording_sequence_status == 2:
            if self.currently_recording == False:
                self.recording_sequence_status = None
                pyautogui.hotkey('F6')
            return -1



    def MonitoringHotkeyCallback(self, inRefcon):
        if self.WindowId == None:
            self.WindowId = xp.createWindowEx(self.windowInfo)
        else:
            xp.destroyWindow(self.WindowId)
            self.WindowId = None
    
    def MoveFileCallback(self, elapsedMe, elapsedSim, counter, refcon):
        data_count = len(os.listdir("C:\\Users\\gabriel\\Desktop\\aircraft-detection\\generated_data"))
        while True:
            try:
                os.mkdir(f'C:\\Users\\gabriel\\Desktop\\aircraft-detection\\generated_data\\{data_count}')
                break
            except:
                data_count+=1

        with open(f'C:\\Users\\gabriel\\Desktop\\aircraft-detection\\generated_data\\{data_count}\\data.pkl', 'wb') as f:
            pkl.dump(self.locs, f)
            
        camera_x = xp.getDataf(self.camera_x)
        camera_y = xp.getDataf(self.camera_y)
        camera_z = xp.getDataf(self.camera_z)
        camera_pitch = xp.getDataf(self.camera_pitch)
        camera_roll = xp.getDataf(self.camera_roll)
        camera_heading = xp.getDataf(self.camera_heading)
        camera_x, camera_y, camera_z = xp.localToWorld(xp.getDataf(self.camera_x), xp.getDataf(self.camera_y), xp.getDataf(self.camera_z))

        with open(f'C:\\Users\\gabriel\\Desktop\\aircraft-detection\\generated_data\\{data_count}\\camera.txt', 'w') as f:
            f.write(f'{camera_x}, {camera_y}, {camera_z}, {camera_pitch}, {camera_roll}, {camera_heading}')

        for i in os.listdir("C:\\X-Plane 11\\Output"):
            if '.avi' in i:
                os.rename(f"C:\\X-Plane 11\\Output\\{i}", f"C:\\Users\\gabriel\\Desktop\\aircraft-detection\\generated_data\\{data_count}\\video.avi")
                break
        for i in os.listdir("C:\\X-Plane 11\\Output"):
            if '.avi' in i:
                os.remove(f'C:\\X-Plane 11\\Output\\{i}')
        return 0

    def LoggingHotKeyCallback(self, inRefcon):
        if self.currently_recording == False:
            self.locs = []
            xp.registerFlightLoopCallback(self.LoggingFlightLoopCallback, -1, 0)
            pyautogui.hotkey('ctrl', 'space')
            self.currently_recording = True

        else:
            pyautogui.hotkey('ctrl', 'space')
            self.currently_recording = False
            xp.unregisterFlightLoopCallback(self.LoggingFlightLoopCallback, 0)
            xp.scheduleFlightLoop(self.DataMovingLoopID, interval = 1)
            
    def LoggingFlightLoopCallback(self, elapsedMe, elapsedSim, counter, refcon):
        elapsed = xp.getElapsedTime()
        self.current_lat = xp.getDataf(self.plane_lat)
        self.current_lon = xp.getDataf(self.plane_lon)
        self.current_el = xp.getDataf(self.plane_el)
        self.locs.append([self.current_lat, self.current_lon, self.current_el])
        if self.current_el < 11 and len(self.locs) > 3:
            # Check if the past 3 elevation is stable (Below 11 and 0.0001 Variance)
            past_three_elevation = [self.locs[-1][2], self.locs[-2][2], self.locs[-3][2]]
            if statistics.variance(past_three_elevation) < 0.00001:
                pyautogui.hotkey('F6')
        return -1   

    def CameraHotKeyCallback_1(self, inRefcon):
        xp.commandOnce(xp.findCommand("sim/view/default_view"))
        xp.controlCamera(xp.ControlCameraUntilViewChanges, self.StaticCamera_1, 0)

    def StaticCamera_1(self, outCameraPosition, inIsLosingControl, inRefcon):
        self.selected_camera = 1
        if (inIsLosingControl):
            xp.dontControlCamera()

        if (outCameraPosition and not inIsLosingControl):
            outCameraPosition[0] = self.camera_static_local_x[0]
            outCameraPosition[1] = self.camera_static_local_y[0]
            outCameraPosition[2] = self.camera_static_local_z[0]
            outCameraPosition[3] = self.camera_static_pitch[0]
            outCameraPosition[4] = self.camera_static_heading[0] 
            outCameraPosition[5] = self.camera_static_roll[0]
            outCameraPosition[6] = self.camera_static_zoom[0]
        return 1
    
    def CameraHotKeyCallback_2(self, inRefcon):
        xp.commandOnce(xp.findCommand("sim/view/default_view"))
        xp.controlCamera(xp.ControlCameraUntilViewChanges, self.StaticCamera_2, 0)

    def StaticCamera_2(self, outCameraPosition, inIsLosingControl, inRefcon):
        self.selected_camera = 2
        if (inIsLosingControl):
            xp.dontControlCamera()

        if (outCameraPosition and not inIsLosingControl):
            outCameraPosition[0] = self.camera_static_local_x[1]
            outCameraPosition[1] = self.camera_static_local_y[1]
            outCameraPosition[2] = self.camera_static_local_z[1]
            outCameraPosition[3] = self.camera_static_pitch[1]
            outCameraPosition[4] = self.camera_static_heading[1]
            outCameraPosition[5] = self.camera_static_roll[1]
            outCameraPosition[6] = self.camera_static_zoom[1]
        return 1

    def DrawWindowCallback(self, inWindowID, inRefcon):
        (left, top, right, bottom) = xp.getWindowGeometry(inWindowID)
        xp.drawTranslucentDarkBox(left, top, right, bottom)
        color = 1.0, 1.0, 1.0
        if self.currently_recording:
            xp.drawString(color, left + 5, top - 20, f'LOGGING MODE (LATLONEL = plane)', 0, xp.Font_Basic)
        else:
            a = xp.localToWorld(xp.getDataf(self.camera_x), xp.getDataf(self.camera_y), xp.getDataf(self.camera_z))
            self.current_lat = a[0]
            self.current_lon = a[1]
            self.current_el = a[2]
            xp.drawString(color, left + 5, top - 20, f'MONITORING MODE (LATLONEL = camera)', 0, xp.Font_Basic)

        #xp.drawString(color, left + 5, top - 40, f'Time     : {xp.getElapsedTime()}', 0, xp.Font_Basic)
        xp.drawString(color, left + 5, top - 40, f'Latitude : {self.current_lat}', 0, xp.Font_Basic)
        xp.drawString(color, left + 5, top - 60, f'Longitude: {self.current_lon}', 0, xp.Font_Basic)
        xp.drawString(color, left + 5, top - 80, f'Elevation: {self.current_el}' , 0, xp.Font_Basic)
        #xp.drawString(color, left + 5, top - 120, f'Pitch    : {camera_pitch}', 0, xp.Font_Basic)
        #xp.drawString(color, left + 5, top - 140, f'Roll     : {camera_roll}', 0, xp.Font_Basic)
        #xp.drawString(color, left + 5, top - 160, f'Heading  : {camera_heading}', 0, xp.Font_Basic)
        xp.drawString(color, left + 5, top - 100, f'Plane Distance  : {geopy.distance.geodesic((1.3541354199301814, 103.97961848373048), (xp.getDataf(self.plane_lat), xp.getDataf(self.plane_lon))).nm}', 0, xp.Font_Basic)
        #xp.drawString(color, left + 5, top - 120, f'recording sequecne: {self.recording_sequence_status, self.selected_camera}' , 0, xp.Font_Basic)

    def KeyCallback(self, inWindowID, inKey, inFlags, inVirtualKey, inRefcon, losingFocus):
        pass

    def MouseClickCallback(self, inWindowID, x, y, inMouse, inRefcon):
        return 1

    def CursorCallback(self, inWindowID, x, y, inRefcon):
        return xp.CursorDefault

    def MouseWheelCallback(self, inWindowID, x, y, wheel, clicks, inRefcon):
        return 1
