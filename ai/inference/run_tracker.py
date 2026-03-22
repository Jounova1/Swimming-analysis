from track_swimmer import SwimmerTracker

tracker = SwimmerTracker(
    min_area=500,
    max_lost=15,
    distance_threshold=80.0
)

observations = tracker.track_video("bf.mp4", visualize=True)

print(f"Total observations: {len(observations)}")
for obs in observations[:10]:
    print(obs)