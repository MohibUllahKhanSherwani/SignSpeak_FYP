import cv2
import os
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime
import json
import winsound

import mediapipe as mp
from mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks

from actions_config import (
    load_actions,
    ACTIONS_FILE,
    DATA_PATH,
    SEQUENCE_LENGTH,
    NUM_SEQUENCES,
    FRAME_WAIT_MS,
)

class DataCollectorGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("SignSpeak - Enhanced Data Collector")
        self.window.geometry("900x600")
        self.window.configure(bg="#2C3E50")
        self.camera_scan_max_index = self._get_camera_scan_max_index()
        
        self.stop_flag = False
        self.pause_flag = False
        self.is_collecting = False
        self.start_time = None
        self.lifetime_seconds = self.load_stats()
        
        try:
            self.actions = load_actions()
            self.current_action = tk.StringVar(value=self.actions[0])
        except (FileNotFoundError, ValueError):
            self.actions = []
            self.current_action = tk.StringVar(value="")

        self.available_camera_indices = self.detect_available_cameras()
        if not self.available_camera_indices:
            self.available_camera_indices = [0]
        self.selected_camera = tk.StringVar(value=str(self.available_camera_indices[0]))
        
        self.build_ui()
        if self.actions:
            self.refresh_table()
        
        self.window.bind("<space>", lambda e: self.toggle_pause())
        self.window.bind("<Escape>", lambda e: self.stop_collection())
        self.window.bind("<s>", lambda e: self.start_collection_thread())
        self.window.bind("<S>", lambda e: self.start_collection_thread())
        
        self.window.mainloop()

    def _backend_candidates(self):
        if os.name == "nt":
            candidates = []
            if hasattr(cv2, "CAP_DSHOW"):
                candidates.append(cv2.CAP_DSHOW)
            if hasattr(cv2, "CAP_MSMF"):
                candidates.append(cv2.CAP_MSMF)
            candidates.append(None)
            return candidates
        return [None]

    def _get_camera_scan_max_index(self):
        value = os.getenv("SIGNSPEAK_CAMERA_MAX_INDEX", "20").strip()
        try:
            return max(0, int(value))
        except ValueError:
            return 20

    def _open_capture(self, index, backend=None):
        if backend is None:
            return cv2.VideoCapture(index)
        return cv2.VideoCapture(index, backend)

    def _camera_has_frames(self, cap, warmup_frames=10):
        if not cap.isOpened():
            return False
        for _ in range(warmup_frames):
            ret, _ = cap.read()
            if ret:
                return True
            time.sleep(0.05)
        return False

    def open_capture_with_fallback(self, index):
        for backend in self._backend_candidates():
            cap = self._open_capture(index, backend)
            if self._camera_has_frames(cap):
                return cap
            cap.release()
        return self._open_capture(index)

    def detect_available_cameras(self, max_index=None):
        if max_index is None:
            max_index = self.camera_scan_max_index
        detected = []
        for idx in range(max_index + 1):
            cap = self.open_capture_with_fallback(idx)
            if self._camera_has_frames(cap, warmup_frames=5):
                detected.append(idx)
            cap.release()
        return detected

    def get_selected_camera_index(self):
        try:
            return int(self.selected_camera.get())
        except (TypeError, ValueError):
            return 0

    def refresh_camera_list(self, show_feedback=True):
        if self.is_collecting:
            return

        detected = self.detect_available_cameras()
        if not detected:
            detected = [0]

        self.available_camera_indices = detected
        values = [str(i) for i in detected]
        self.camera_dropdown.config(values=values)

        if self.selected_camera.get() not in values:
            self.selected_camera.set(values[0])

        if show_feedback:
            messagebox.showinfo("Camera Scan", f"Detected camera indices: {', '.join(values)}")

    def build_ui(self):
        header_frame = tk.Frame(self.window, bg="#34495E", height=60)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="SignSpeak Data Collection Studio",
            font=("Arial", 20, "bold"),
            bg="#34495E",
            fg="#ECF0F1"
        ).pack(pady=10)
        
        self.action_display_frame = tk.Frame(self.window, bg="#1ABC9C", height=80)
        self.action_display_frame.pack(fill=tk.X, padx=10, pady=5)
        self.action_display_frame.pack_propagate(False)
        
        tk.Label(
            self.action_display_frame,
            text="Current Sign:",
            font=("Arial", 12),
            bg="#1ABC9C",
            fg="#FFFFFF"
        ).pack(pady=(5, 0))
        
        self.current_action_label = tk.Label(
            self.action_display_frame,
            text="No action selected",
            font=("Arial", 28, "bold"),
            bg="#1ABC9C",
            fg="#FFFFFF"
        )
        self.current_action_label.pack()
        control_frame = tk.Frame(self.window, bg="#2C3E50")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        left_controls = tk.Frame(control_frame, bg="#2C3E50")
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            left_controls,
            text="Select Action:",
            font=("Arial", 12),
            bg="#2C3E50",
            fg="#ECF0F1"
        ).grid(row=0, column=0, padx=5, sticky="w")
        
        self.dropdown = ttk.Combobox(
            left_controls,
            values=self.actions,
            textvariable=self.current_action,
            width=25,
            font=("Arial", 11)
        )
        self.dropdown.grid(row=0, column=1, padx=5)
        self.dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_action_display())
        
        tk.Button(
            left_controls,
            text="Add Action",
            command=self.add_action,
            bg="#3AAFA9",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        ).grid(row=0, column=2, padx=5)
        
        tk.Button(
            left_controls,
            text="Remove",
            command=self.remove_action,
            bg="#E74C3C",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        ).grid(row=0, column=3, padx=5)

        tk.Label(
            left_controls,
            text="Camera Index:",
            font=("Arial", 12),
            bg="#2C3E50",
            fg="#ECF0F1"
        ).grid(row=1, column=0, padx=5, pady=(10, 0), sticky="w")

        self.camera_dropdown = ttk.Combobox(
            left_controls,
            values=[str(i) for i in self.available_camera_indices],
            textvariable=self.selected_camera,
            width=10,
            font=("Arial", 11),
            state="readonly"
        )
        self.camera_dropdown.grid(row=1, column=1, padx=5, pady=(10, 0), sticky="w")

        tk.Button(
            left_controls,
            text="Rescan Cameras",
            command=self.refresh_camera_list,
            bg="#3498DB",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        ).grid(row=1, column=2, columnspan=2, padx=5, pady=(10, 0), sticky="w")
        
        stats_frame = tk.Frame(self.window, bg="#34495E")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.session_time_label = tk.Label(
            stats_frame,
            text="Session: 00:00:00",
            font=("Arial", 11),
            bg="#34495E",
            fg="#ECF0F1"
        )
        self.session_time_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.lifetime_time_label = tk.Label(
            stats_frame,
            text="Total Collection: 00:00:00",
            font=("Arial", 11),
            bg="#34495E",
            fg="#ECF0F1"
        )
        self.lifetime_time_label.pack(side=tk.LEFT, padx=20, pady=10)        
        table_frame = tk.Frame(self.window, bg="#2C3E50")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(
            table_frame,
            text="Actions Progress",
            font=("Arial", 13, "bold"),
            bg="#2C3E50",
            fg="#ECF0F1"
        ).pack(anchor="w", pady=(0, 5))
        
        # Scrollbar for table
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        columns = ("action", "collected", "needed", "percentage")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=5,
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.tree.yview)
        
        self.tree.heading("action", text="Action Name")
        self.tree.heading("collected", text="Collected")
        self.tree.heading("needed", text="Target")
        self.tree.heading("percentage", text="Progress")
        
        self.tree.column("action", anchor="w", width=200)
        self.tree.column("collected", anchor="center", width=100)
        self.tree.column("needed", anchor="center", width=100)
        self.tree.column("percentage", anchor="center", width=120)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.tree_menu = tk.Menu(self.tree, tearoff=0)
        self.tree_menu.add_command(label="Delete Last Sequence", command=self.delete_last_sequence)
        self.tree.bind("<Button-3>", self.show_tree_menu)
        
        progress_frame = tk.Frame(self.window, bg="#34495E")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(
            progress_frame,
            text="Ready to start",
            font=("Arial", 14, "bold"),
            bg="#34495E",
            fg="#ECF0F1"
        )
        self.status_label.pack(pady=(10, 5))
        
        self.progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=600,
            mode="determinate"
        )
        self.progress.pack(pady=5)
        
        self.progress_detail = tk.Label(
            progress_frame,
            text="0 / 50 sequences",
            font=("Arial", 11),
            bg="#34495E",
            fg="#BDC3C7"
        )
        self.progress_detail.pack(pady=(0, 10))
        button_frame = tk.Frame(self.window, bg="#2C3E50")
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = tk.Button(
            button_frame,
            text="START COLLECTING",
            command=self.start_collection_thread,
            bg="#27AE60",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=30,
            pady=12,
            relief=tk.RAISED,
            bd=3
        )
        self.start_btn.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.pause_btn = tk.Button(
            button_frame,
            text="PAUSE",
            command=self.toggle_pause,
            bg="#F39C12",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=30,
            pady=12,
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="STOP",
            command=self.stop_collection,
            bg="#E74C3C",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=30,
            pady=12,
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=10, expand=True)
        
        shortcuts_frame = tk.Frame(self.window, bg="#34495E")
        shortcuts_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(
            shortcuts_frame,
            text="Shortcuts: [SPACE] Pause/Resume  |  [ESC] Stop  |  [Q] in camera window to quit",
            font=("Arial", 9),
            bg="#34495E",
            fg="#95A5A6"
        ).pack(pady=5)
        
        if self.actions:
            self.update_action_display()
            
    def update_action_display(self):
        action = self.current_action.get()
        if action:
            self.current_action_label.config(text=action.upper().replace("_", " "))
        else:
            self.current_action_label.config(text="No action selected")

    def show_tree_menu(self, event):
        try:
            self.tree.selection_set(self.tree.identify_row(event.y))
            self.tree_menu.post(event.x_root, event.y_root)
        finally:
            self.tree_menu.grab_release()

    def delete_last_sequence(self):
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        action = item['values'][0]
        
        collected = self.count_sequences(action)
        if collected == 0:
            messagebox.showwarning("No Data", f"No sequences found for '{action}'")
            return
        
        if messagebox.askyesno("Delete Sequence", 
                              f"Delete last sequence ({collected-1}) for '{action}'?"):
            import shutil
            seq_folder = os.path.join(DATA_PATH, action, str(collected - 1))
            if os.path.exists(seq_folder):
                shutil.rmtree(seq_folder)
                self.refresh_table()
                messagebox.showinfo("Deleted", f"Sequence {collected-1} deleted for '{action}'")
    def add_action(self):
        new_action = simpledialog.askstring("Add Action", "Enter new action name:")
        if new_action and new_action.strip():
            new_action = new_action.lower().replace(" ", "_")
            if new_action in self.actions:
                messagebox.showwarning("Duplicate", f"Action '{new_action}' already exists!")
                return

            with open(ACTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(new_action + "\n")

            self.actions = load_actions()
            self.dropdown.config(values=self.actions)
            self.current_action.set(new_action)
            self.refresh_table()
            self.update_action_display()

            messagebox.showinfo("Added", f"Action '{new_action}' added successfully!")

    def remove_action(self):
        action = self.current_action.get()
        if not action:
            messagebox.showwarning("No Selection", "Please select an action to remove")
            return
            
        if messagebox.askyesno("Remove Action", 
                              f"Delete '{action}' from actions list?\n\nNote: Collected data will NOT be deleted."):
            self.actions = [a for a in self.actions if a != action]

            with open(ACTIONS_FILE, "w", encoding="utf-8") as f:
                for a in self.actions:
                    f.write(a + "\n")

            self.dropdown.config(values=self.actions)
            if self.actions:
                self.current_action.set(self.actions[0])
            else:
                self.current_action.set("")

            self.refresh_table()
            self.update_action_display()
            messagebox.showinfo("Removed", f"Action '{action}' removed from list.")

    def count_sequences(self, action):
        folder = os.path.join(DATA_PATH, action)
        if not os.path.exists(folder):
            return 0
        return len([seq for seq in os.listdir(folder) if seq.isdigit()])

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        total_collected = 0
        total_needed = 0

        for action in self.actions:
            collected = self.count_sequences(action)
            needed = NUM_SEQUENCES
            percentage = int((collected / needed) * 100) if needed > 0 else 0
            
            total_collected += collected
            
            self.tree.insert("", tk.END, values=(
                action,
                collected,
                needed,
                f"{percentage}%"
            ))
        
        l_hours, l_remainder = divmod(self.lifetime_seconds, 3600)
        l_minutes, l_seconds = divmod(l_remainder, 60)
        self.lifetime_time_label.config(text=f"🌍 Total Collection: {l_hours:02d}:{l_minutes:02d}:{l_seconds:02d}")
    def update_session_timer(self):
        if self.start_time and self.is_collecting:
            elapsed = datetime.now() - self.start_time
            s_hours, s_remainder = divmod(int(elapsed.total_seconds()), 3600)
            s_minutes, s_seconds = divmod(s_remainder, 60)
            self.session_time_label.config(text=f"⏱️ Session: {s_hours:02d}:{s_minutes:02d}:{s_seconds:02d}")
            
            self.lifetime_seconds += 1
            l_hours, l_remainder = divmod(self.lifetime_seconds, 3600)
            l_minutes, l_seconds = divmod(l_remainder, 60)
            self.lifetime_time_label.config(text=f" Total Collection: {l_hours:02d}:{l_minutes:02d}:{l_seconds:02d}")
            if self.lifetime_seconds % 60 == 0:
                self.save_stats()
            
            if self.is_collecting:
                self.window.after(1000, self.update_session_timer)

    def load_stats(self):
        stats_file = os.path.join(DATA_PATH, "stats.json")
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                    return data.get("lifetime_seconds", 0)
            except:
                return 0
        return 0

    def save_stats(self):
        os.makedirs(DATA_PATH, exist_ok=True)
        stats_file = os.path.join(DATA_PATH, "stats.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump({"lifetime_seconds": self.lifetime_seconds}, f)
        except:
            pass
    def start_collection_thread(self):
        if not self.current_action.get():
            messagebox.showwarning("No Action", "Please select an action first!")
            return
            
        self.is_collecting = True
        self.start_time = datetime.now()
        
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.dropdown.config(state=tk.DISABLED)
        self.camera_dropdown.config(state=tk.DISABLED)
        
        self.update_session_timer()
        
        thread = threading.Thread(target=self.collect)
        thread.daemon = True
        thread.start()

    def toggle_pause(self):
        if not self.is_collecting:
            return
            
        self.pause_flag = not self.pause_flag
        
        if self.pause_flag:
            self.pause_btn.config(text="RESUME", bg="#27AE60")
            self.status_label.config(text="PAUSED", fg="#F39C12")
        else:
            self.pause_btn.config(text="PAUSE", bg="#F39C12")
            self.status_label.config(text="RECORDING", fg="#E74C3C")

    def stop_collection(self):
        if not self.is_collecting:
            return
            
        self.stop_flag = True
        self.is_collecting = False
        self.status_label.config(text="Stopping...", fg="#95A5A6")
        
    def beep(self, frequency=1000, duration=100):
        try:
            winsound.Beep(frequency, duration)
        except:
            pass

    def collect(self):
        action = self.current_action.get()
        self.stop_flag = False
        self.pause_flag = False

        self.create_folders(action)

        sequences_to_collect = self.get_missing_sequences(action)
        if not sequences_to_collect:
            messagebox.showinfo("Complete", f"All {NUM_SEQUENCES} sequences already recorded for {action}")
            self.reset_ui_after_collection()
            return
        total = NUM_SEQUENCES
        completed_count = NUM_SEQUENCES - len(sequences_to_collect)
        self.progress["maximum"] = total
        self.progress["value"] = completed_count
        camera_index = self.get_selected_camera_index()
        cap = self.open_capture_with_fallback(camera_index)
        
        if not self._camera_has_frames(cap, warmup_frames=10):
            messagebox.showerror(
                "Camera Error",
                f"Could not open camera index {camera_index}.\n"
                "Click 'Rescan Cameras' and try a different index."
            )
            cap.release()
            self.reset_ui_after_collection()
            return

        with mp.solutions.holistic.Holistic(min_detection_confidence=0.4,
                                  min_tracking_confidence=0.4) as holistic:

            for seq in sequences_to_collect:
                if self.stop_flag:
                    break
                for i in range(1, 0, -1):
                    if self.stop_flag:
                        break
                    self.status_label.config(text=f"Get Ready! Starting in {i}...", fg="#F39C12")
                    self.progress_detail.config(text=f"Sequence {seq+1}/{NUM_SEQUENCES}")
                    self.beep(800, 150)
                    time.sleep(1)                    
                    while self.pause_flag and not self.stop_flag:
                        time.sleep(0.1)

                if self.stop_flag:
                    break

                self.beep(1200, 200)
                self.status_label.config(text=f"RECORDING Sequence {seq+1}/{NUM_SEQUENCES}", fg="#E74C3C")

                seq_folder = os.path.join(DATA_PATH, action, str(seq))
                os.makedirs(seq_folder, exist_ok=True)

                for frame_num in range(SEQUENCE_LENGTH):
                    if self.stop_flag:
                        break
                        
                    while self.pause_flag and not self.stop_flag:
                        time.sleep(0.1)

                    ret, frame = cap.read()
                    if not ret:
                        self.status_label.config(text="Camera error!", fg="#E74C3C")
                        self.stop_flag = True
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)                    
                    cv2.putText(image, f"Frame: {frame_num+1}/{SEQUENCE_LENGTH}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Seq: {seq+1}/{NUM_SEQUENCES}",
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, action.upper().replace("_", " "),
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

                    key = extract_keypoints(results)
                    np.save(os.path.join(seq_folder, f"{frame_num}.npy"), key)

                    cv2.imshow("SignSpeak - Recording", image)
                    if cv2.waitKey(FRAME_WAIT_MS) & 0xFF == ord('q'):
                        self.stop_flag = True
                        break
                if not self.stop_flag:
                    self.beep(1500, 150)
                    
                self.progress["value"] = seq + 1
                percent = int(((seq + 1) / total) * 100)
                self.progress_detail.config(text=f"{seq+1} / {total} sequences ({percent}%)")

                self.refresh_table()

        cap.release()
        cv2.destroyAllWindows()

        self.refresh_table()

        if self.stop_flag:
            self.status_label.config(text="Stopped", fg="#95A5A6")
        else:
            self.status_label.config(text="Completed!", fg="#27AE60")
            self.beep(1000, 300)
            messagebox.showinfo("Done", f"All {NUM_SEQUENCES} sequences recorded for '{action}'!")
        self.reset_ui_after_collection()

    def reset_ui_after_collection(self):
        self.is_collecting = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="PAUSE", bg="#F39C12")
        self.stop_btn.config(state=tk.DISABLED)
        self.dropdown.config(state=tk.NORMAL)
        self.camera_dropdown.config(state="readonly")
        self.save_stats()

    def create_folders(self, action):
        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path, exist_ok=True)
    def get_missing_sequences(self, action):
        folder = os.path.join(DATA_PATH, action)
        if not os.path.exists(folder):
            return list(range(NUM_SEQUENCES)) 
            
        existing = set()
        for seq in os.listdir(folder):
            if seq.isdigit():
                existing.add(int(seq))
                
        missing = []
        for i in range(NUM_SEQUENCES):
            if i not in existing:
                missing.append(i)
        return sorted(missing)


if __name__ == "__main__":
    DataCollectorGUI()
