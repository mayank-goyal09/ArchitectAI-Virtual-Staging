from src.processor import RoomProcessor

# Initialize with much lower thresholds for bright rooms
# low_threshold=20, high_threshold=50 (instead of 50/150)
engine = RoomProcessor(low_threshold=20, high_threshold=70)

# RUN IT! (Make sure you have an image named 'test_room.jpg' in your folder)
skeleton = engine.extract_structure('test_room.jpg')
engine.save_map(skeleton, 'data/processed/room_skeleton_v2.png')