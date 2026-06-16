from PyQt6.QtCore import QObject, pyqtSignal, QTimer
import os
import uuid

from core.processor import VideoProcessorThread
from core.filters import BeautyFilterEngine

class BatchQueueItem:
    """
    Standard data container representing a video processing job in the queue.
    """
    def __init__(self, input_path, output_path, codec_name, params, preset_name="Custom"):
        self.id = str(uuid.uuid4())[:8]  # Unique short ID
        self.input_path = input_path
        self.output_path = output_path
        self.codec_name = codec_name
        self.params = params.copy()
        self.preset_name = preset_name
        self.status = "Queued"  # Queued, Processing, Completed, Failed
        self.progress = 0.0
        self.eta = ""
        self.error_message = ""

    def get_display_name(self):
        return os.path.basename(self.input_path)


class BatchQueueProcessor(QObject):
    """
    Controller that manages queue items and sequences background exports.
    """
    item_added = pyqtSignal(BatchQueueItem)
    item_removed = pyqtSignal(str)  # item_id
    item_updated = pyqtSignal(BatchQueueItem)
    queue_status_changed = pyqtSignal(bool)  # is_processing
    queue_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.items = []
        self.active_thread = None
        self.active_item = None
        self._is_processing = False

    def add_item(self, input_path, output_path, codec_name, params, preset_name="Custom"):
        """
        Adds a new item to the batch queue.
        """
        item = BatchQueueItem(input_path, output_path, codec_name, params, preset_name)
        self.items.append(item)
        self.item_added.emit(item)
        return item

    def remove_item(self, item_id):
        """
        Removes an item from the queue. If it is active, stops execution first.
        """
        # Find item
        target_item = None
        for item in self.items:
            if item.id == item_id:
                target_item = item
                break

        if not target_item:
            return False

        # If it's active, stop processing first
        if self.active_item and self.active_item.id == item_id:
            self.pause_queue()

        self.items.remove(target_item)
        self.item_removed.emit(item_id)
        return True

    def clear_completed(self):
        """
        Removes all completed and failed items from the list.
        """
        to_remove = [item.id for item in self.items if item.status in ("Completed", "Failed")]
        for item_id in to_remove:
            self.remove_item(item_id)

    def start_queue(self):
        """
        Starts sequential background queue processing.
        """
        if self._is_processing:
            return
        
        # Check if we have anything to process
        has_queued = any(item.status == "Queued" for item in self.items)
        if not has_queued:
            return
            
        self._is_processing = True
        self.queue_status_changed.emit(True)
        self._process_next()

    def pause_queue(self):
        """
        Pauses queue execution and stops active worker threads.
        """
        if not self._is_processing:
            return
            
        self._is_processing = False
        self.queue_status_changed.emit(False)

        if self.active_thread and self.active_thread.isRunning():
            self.active_thread.stop()
            self.active_thread.wait()

        if self.active_item:
            self.active_item.status = "Queued"
            self.active_item.progress = 0.0
            self.active_item.eta = ""
            self.item_updated.emit(self.active_item)
            
        self.active_item = None
        self.active_thread = None

    def is_processing(self):
        return self._is_processing

    def _process_next(self):
        if not self._is_processing:
            return

        # Find first queued item
        next_item = None
        for item in self.items:
            if item.status == "Queued":
                next_item = item
                break

        if not next_item:
            # Completed everything queued
            self._is_processing = False
            self.queue_status_changed.emit(False)
            self.queue_finished.emit()
            return

        self.active_item = next_item
        self.active_item.status = "Processing"
        self.active_item.progress = 0.0
        self.active_item.eta = "Calculating..."
        self.item_updated.emit(self.active_item)

        try:
            # Instantiate isolated filter engine to avoid main thread collisions
            filter_engine = BeautyFilterEngine()
            
            self.active_thread = VideoProcessorThread(
                input_path=next_item.input_path,
                output_path=next_item.output_path,
                codec_name=next_item.codec_name,
                filter_engine=filter_engine,
                params=next_item.params
            )
            self.active_thread.progress.connect(self._on_thread_progress)
            self.active_thread.completed.connect(self._on_thread_completed)
            self.active_thread.error.connect(self._on_thread_error)
            self.active_thread.start()
        except Exception as e:
            self._on_thread_error(str(e))

    def _on_thread_progress(self, frame_idx, percentage, eta_str):
        if self.active_item:
            self.active_item.progress = percentage
            self.active_item.eta = eta_str
            self.item_updated.emit(self.active_item)

    def _on_thread_completed(self, output_path, use_fallback):
        if self.active_item:
            self.active_item.status = "Completed"
            self.active_item.progress = 100.0
            self.active_item.eta = "Done"
            if use_fallback:
                self.active_item.error_message = "Completed with fallback H.264 MP4 codec."
            self.item_updated.emit(self.active_item)

        self.active_item = None
        self.active_thread = None

        # Schedule next item processing
        QTimer.singleShot(100, self._process_next)

    def _on_thread_error(self, err_msg):
        if self.active_item:
            self.active_item.status = "Failed"
            self.active_item.progress = 0.0
            self.active_item.eta = ""
            self.active_item.error_message = err_msg
            self.item_updated.emit(self.active_item)

        self.active_item = None
        self.active_thread = None

        # Schedule next item processing (so one error doesn't halt the entire queue)
        QTimer.singleShot(100, self._process_next)
