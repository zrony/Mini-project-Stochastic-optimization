import pandas as pd
import random
import copy


def longest_subsequence_nlogn(coordinates):
    n = len(coordinates)
    # Sort the coordinates based on x values in descending order, if tied, then by y in descending order
    sorted_coords = sorted(coordinates, key=lambda coord: (coord[0], coord[1]), reverse=True)

    # Initialize dynamic programming list and ends list
    dp = [0] * n
    ends = [0] * n
    dp[0] = 1
    ends[0] = sorted_coords[0]
    length = 1

    for i in range(1, n):
        # Binary search to find the position of the current coordinate in the ends list
        left, right = 0, length
        while left < right:
            mid = (left + right) // 2
            if ends[mid][1] <= sorted_coords[i][1]:
                right = mid
            else:
                left = mid + 1

        dp[i] = left + 1
        ends[left] = sorted_coords[i]

        if left == length:
            length += 1

    # Backtrack to find the longest subsequence
    longest_subseq = []
    curr_len = length
    for i in reversed(range(n)):
        if dp[i] == curr_len:
            longest_subseq.append(sorted_coords[i])
            curr_len -= 1

    return length, longest_subseq[::-1]


def heaviest_increasing_subsequence(coordinates):
    # Sort the list based on x and then y
    sorted_coords = sorted(coordinates, key=lambda x: (x[0], x[1]))

    n = len(sorted_coords)
    dp = [0] * n
    prev = [-1] * n  # store predecessors

    # Initialize the dp array with weights of tuples
    for i in range(n):
        dp[i] = sorted_coords[i][2]

    # Iterate over all tuples to find the maximum weight
    for i in range(n):
        for j in range(i):
            if sorted_coords[j][0] < sorted_coords[i][0] and sorted_coords[j][1] < sorted_coords[i][1]:
                if dp[j] + sorted_coords[i][2] > dp[i]:
                    dp[i] = dp[j] + sorted_coords[i][2]
                    prev[i] = j 

    # Find the index with the maximum weight
    max_idx = dp.index(max(dp))

    # Traceback the path
    path = []
    while max_idx != -1:
        path.append(sorted_coords[max_idx])
        max_idx = prev[max_idx]
    path.reverse()

    return max(dp), path

# first algorithm
def update_weights_randomly(coordinates):
    # Finding the longest subsequence of the array
    W, longest_subseq = longest_subsequence_nlogn(coordinates)
    sub_array = []

    # remove the longest subsequence from the optional set of points to check
    possible_points = copy.copy(coordinates)
    for point in longest_subseq:
        possible_points.remove(point)
    changeable_points = set(possible_points)

    # choose a point randomly and if it's not part of the heaviest path - change its weight to 2
    while changeable_points:
        random_point = random.choice(list(changeable_points))
        index = coordinates.index(random_point)
        coordinates[index] = (random_point[0], random_point[1], 2)
        new_W, sub = finding_heaviest_subsequence(coordinates)
        if new_W <= W:
            sub_array.append(coordinates[index])
        else:
            coordinates[index] = (random_point[0], random_point[1], 1)
        changeable_points.remove(random_point)

    return len(sub_array), sub_array

# second algorithm
def update_weights_iterative(coordinates):
    # finding the longest subsequence of the array
    W, longest_subseq = longest_subsequence_nlogn(coordinates)
    sub_array = []

    # remove the longest subsequence from the optional set of point to check
    possible_points = copy.copy(coordinates)
    for point in longest_subseq:
        possible_points.remove(point)
    changeable_points = set(possible_points)

    # check every point in iterative order, and if it's not part of the heaviest path - change its weight to 2
    count = 0
    for point in changeable_points:
        count +=1
        index = coordinates.index(point)
        coordinates[index] = (point[0], point[1], 2)
        new_W, sub= heaviest_increasing_subsequence(coordinates)
        if count % 10 == 0:
            print(count)
        if new_W <= W:
            sub_array.append(coordinates[index])
        else:
            coordinates[index] = (point[0], point[1], 1)

    return len(sub_array), sub_array

# third algorithm
def find_coordinate_with_largest_difference(coordinates):
    largest_difference = float('-inf')
    result_coordinate = None

    for coord in coordinates:
        x, y = coord[0], coord[1]
        difference = abs(x - y)
        if difference > largest_difference:
            largest_difference = difference
            result_coordinate = coord

    return result_coordinate


def update_weights_intelligence_algorithm(coordinates):
    W, longest_subseq = longest_subsequence_nlogn(coordinates)
    # Insert all the coordinates into the 'changeablePoints' set
    changeablePoints = set(coordinates)
    # Initialize the 'changedPoints' set to keep track of points with updated weights
    changedPoints = set()
    count = 0
    # Iterate until the 'changeablePoints' set is empty
    while changeablePoints:
        count +=1
        # Choose the point with one of the methods
        target_coord = find_coordinate_with_largest_difference(changeablePoints)
        index = coordinates.index(target_coord)
        updated_coord = target_coord[:2] + (2,)  # increase weight
        coordinates[index] = updated_coord  # update coordinate
        new_weight, chain = heaviest_increasing_subsequence(coordinates)
        if count % 10 == 0:
            print(count)

        # Check if the new weight is heavier than the heaviest chain
        if new_weight > W:
            # break, return to the original state
            coordinates[index] = target_coord
        else:
            changedPoints.add(updated_coord)  # We can change the point's weight.

        # Remove the checked point from the 'changeablePoints' set
        changeablePoints.remove(target_coord)  # Remove the target coordinate from the set

    return len(changedPoints), changedPoints

# get coordinates from the Excel
def convert_excel_to_array(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    x = df['Column 1'].values
    y = df['Column 2'].values

    coordinates = []

    for i in range(len(x)):
        coordinates.append((x[i], y[i], 1))

    return coordinates

