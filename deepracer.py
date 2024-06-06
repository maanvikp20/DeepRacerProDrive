import math

def reward_function(params):
    '''
    Advanced reward function for smoother navigation with predictive steering and
    adaptive speed control based on track curvature.
    '''

    # Extract parameters
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    speed = params['speed']
    steering_angle = params['steering_angle']

    # Constants
    DESIRED_DISTANCE_FROM_CENTER = track_width * 0.25
    MAX_SPEED = 4.0
    STEERING_SENSITIVITY = 15
    TURN_SENSITIVITY = 1.5

    # Calculate the immediate track direction
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction) % 360

    # Direction difference penalty
    direction_diff = abs(track_direction - heading)
    direction_diff = min(direction_diff, 360 - direction_diff)
    direction_penalty = max(1.0 - (direction_diff / 50.0), 0)

    # Distance from center penalty
    distance_penalty = max(1.0 - (distance_from_center / (track_width / 2)), 0)

    # Look-ahead to estimate future curvature
    look_ahead_points = waypoints[closest_waypoints[1]:closest_waypoints[1]+5]  # Look ahead 5 points
    cumulative_turn = 0
    for i in range(1, len(look_ahead_points)):
        # Calculate direction for each segment
        segment_direction = math.atan2(look_ahead_points[i][1] - look_ahead_points[i-1][1],
                                       look_ahead_points[i][0] - look_ahead_points[i-1][0])
        segment_direction = math.degrees(segment_direction) % 360
        if i == 1:
            previous_direction = segment_direction
            continue
        # Accumulate the change in direction
        direction_change = abs(segment_direction - previous_direction)
        direction_change = min(direction_change, 360 - direction_change)
        cumulative_turn += direction_change
        previous_direction = segment_direction

    # Adjust steering based on cumulative turn in the look-ahead window
    if cumulative_turn > 30:  # Indicates a series of turns or a sharp turn
        steering_adjustment = min(cumulative_turn / TURN_SENSITIVITY, 1.0)
    else:
        steering_adjustment = 1.0

    # Adjust speed based on anticipated track curvature
    if cumulative_turn > 30:
        speed_adjustment = 0.5  # Slow down for sharp turns
    else:
        speed_adjustment = 1.0  # Maintain or accelerate on straighter segments

    # Combine rewards and adjustments
    reward = direction_penalty * distance_penalty
    reward *= steering_adjustment * speed_adjustment

    # Discourage going off-track
    if distance_from_center >= track_width / 2:
        reward = 1e-3

    return float(reward)
