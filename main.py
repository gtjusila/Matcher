import streamlit as st
import random
import pandas as pd
import networkx as nx

def initialize_state():
    if 'tournament_state' not in st.session_state:
        st.session_state.tournament_state = {
            'page': 1,
            'seed': 42,
            'n': 10,
            'rounds': 3,
            'current_round': 1,  # Track the current round
            'teams': [],
            'matches': [],
            'validation_error': False,
        }

initialize_state()
tournament_state = st.session_state.tournament_state

def next_page():
    tournament_state['page'] += 1
    st.rerun()

def next_round():
    tournament_state['current_round'] += 1
    if tournament_state['current_round'] > tournament_state['rounds']:
        tournament_state['page'] = 5  # Move to a final results page or summary
    else:
        tournament_state['page'] = 2  # Start the next round
    st.rerun()

def compute_leaderboard(matches):
    leaderboard = {}

    for match in matches:
        team1 = match['team1']
        team2 = match['team2']
        score1 = match['score1']
        score2 = match['score2']

        if team1 not in leaderboard:
            leaderboard[team1] = {"Wins": 0, "Draws": 0, "Losses": 0, "Goals For": 0, "Goals Against": 0, "Goal Difference": 0, "Points": 0}
        if team2 not in leaderboard:
            leaderboard[team2] = {"Wins": 0, "Draws": 0, "Losses": 0, "Goals For": 0, "Goals Against": 0, "Goal Difference": 0, "Points": 0}

        if match['round'] == tournament_state['current_round']:
            continue

        leaderboard[team1]["Goals For"] += score1
        leaderboard[team1]["Goals Against"] += score2
        leaderboard[team2]["Goals For"] += score2
        leaderboard[team2]["Goals Against"] += score1

        leaderboard[team1]["Goal Difference"] = leaderboard[team1]["Goals For"] - leaderboard[team1]["Goals Against"]
        leaderboard[team2]["Goal Difference"] = leaderboard[team2]["Goals For"] - leaderboard[team2]["Goals Against"]

        if score1 > score2:
            leaderboard[team1]["Wins"] += 1
            leaderboard[team2]["Losses"] += 1
        elif score1 < score2:
            leaderboard[team2]["Wins"] += 1
            leaderboard[team1]["Losses"] += 1
        else:
            leaderboard[team1]["Draws"] += 1
            leaderboard[team2]["Draws"] += 1

        leaderboard[team1]["Points"] = leaderboard[team1]["Wins"] * 3 + leaderboard[team1]["Draws"]
        leaderboard[team2]["Points"] = leaderboard[team2]["Wins"] * 3 + leaderboard[team2]["Draws"]
    
    leaderboard_df = pd.DataFrame(leaderboard).T
    leaderboard_df = leaderboard_df.sort_values(by=["Points", "Goal Difference", "Goals For"], ascending=[False, False, False ])

    return leaderboard_df


def decide_matchup(teams, round_num, past_matches, leaderboard):
    """Decide the matchups for a round considering past matchups and the leaderboard."""

    # Step 1: Create the team data structure
    def create_team_data_structure(teams, matches):
        """Create a list of dictionaries containing teamname, points, and past opposition."""
        team_data = []

        for team in teams:
            team_data.append({
                'teamname': team,
                'points': 0,
                'past_opposition': []
            })

        for match in matches:
            team1_data = next(item for item in team_data if item['teamname'] == match['team1'])
            team2_data = next(item for item in team_data if item['teamname'] == match['team2'])

            if match['score1'] > match['score2']:
                team1_data['points'] += 3  # Team 1 wins
            elif match['score1'] < match['score2']:
                team2_data['points'] += 3  # Team 2 wins
            else:
                team1_data['points'] += 1  # Draw
                team2_data['points'] += 1  # Draw

            team1_data['past_opposition'].append(match['team2'])
            team2_data['past_opposition'].append(match['team1'])

        return team_data

    # Generate the team data structure based on past matches
    team_data_structure = create_team_data_structure(teams, past_matches)

    # Step 2: Group teams by points
    groups = {}
    for team in team_data_structure:
        points = team['points']
        if points not in groups:
            groups[points] = []
        groups[points].append(team)

    # Step 3: Match teams within groups using maximum matching
    new_matches = []
    sorted_groups = sorted(groups.items(), key=lambda x: -x[0])  # Sort by points descending
    priority_teams = set()  # Keep track of demoted teams
    priority_level = 1  # Start with 1 for the top group

    for idx, (points, group) in enumerate(sorted_groups):
        G = nx.Graph()

        # Add nodes for all teams in the group
        for team in group:
            G.add_node(team['teamname'])

        # Print the list of all nodes (teams) at this stage
        print(f"Stage: Group with {points} points, Priority Level {priority_level}")
        print("Nodes:", list(G.nodes))

        # Add edges for teams that haven't played against each other
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[j]['teamname'] not in group[i]['past_opposition']:
                    if group[i]['teamname'] in priority_teams or group[j]['teamname'] in priority_teams:
                        # Higher priority for demoted teams with 10^priority_level weight
                        G.add_edge(group[i]['teamname'], group[j]['teamname'], weight=10)
                    else:
                        G.add_edge(group[i]['teamname'], group[j]['teamname'], weight=1)

        # Solve maximum matching with prioritization
        matching = nx.max_weight_matching(G, maxcardinality=True)

        matched_teams = set()
        for team1, team2 in matching:
            new_matches.append({
                'round': round_num,
                'team1': team1,
                'team2': team2,
                'score1': 0,
                'score2': 0
            })
            matched_teams.update([team1, team2])

        # Move unmatched teams to the next group
        unmatched_teams = []
        for team in group:
            if team['teamname'] not in matched_teams:
                unmatched_teams.append(team)
                priority_teams.add(team['teamname'])  # Mark team as demoted for prioritization

        # Add unmatched teams to the next group
        if unmatched_teams and idx + 1 < len(sorted_groups):
            next_group_points = sorted_groups[idx + 1][0]
            groups[next_group_points].extend(unmatched_teams)

    # If there are any unmatched teams left after all groups are processed, shuffle them and pair them randomly
    if unmatched_teams:
        random.shuffle(unmatched_teams)
        for i in range(0, len(unmatched_teams), 2):
            if i + 1 < len(unmatched_teams):
                new_matches.append({
                    'round': round_num,
                    'team1': unmatched_teams[i]['teamname'],
                    'team2': unmatched_teams[i + 1]['teamname'],
                    'score1': 0,
                    'score2': 0
                })

    # Debug print to check the matchups
    return new_matches

# Page 1: Configure Tournament
def page_1():
    st.title("Tournament Setup")
    st.header("Configure Tournament")

    tournament_state['seed'] = st.number_input("Enter a random seed:", value=tournament_state['seed'])
    
    n = st.number_input("Select the number of teams (even only):", min_value=2, max_value=16, value=tournament_state['n'], step=2)
    
    if n != tournament_state['n']:
        tournament_state['n'] = n
        st.rerun()

    rounds = st.number_input("Enter the number of rounds:", min_value=1, value=tournament_state['rounds'], step=1)
    
    if rounds != tournament_state['rounds']:
        tournament_state['rounds'] = rounds
        st.rerun()

    st.subheader("Enter Team Names:")
    default_names = [f"Team {chr(65 + i)}" for i in range(tournament_state['n'])]
    team_names = []
    for i in range(tournament_state['n']):
        team_name = st.text_input(f"Team {i + 1} Name:", value=default_names[i])
        team_names.append(team_name)
    
    tournament_state['teams'] = team_names

    if st.button("Next"):
        random.seed(tournament_state['seed'])
        tournament_state['current_round'] = 1
        next_page()

def page_2():
    st.title(f"Tournament Round {tournament_state['current_round']} Setup")
    st.header("Match Setup")

    is_decided = st.checkbox(f"Is the Round {tournament_state['current_round']} match decided?", value=False)
    
    if is_decided:
        st.subheader("Select Teams for Each Match")
        matches = []

        for i in range(0, tournament_state['n'], 2):
            col1, col2 = st.columns(2)

            with col1:
                team1 = st.selectbox(f"Match {i//2 + 1} - Team 1", tournament_state['teams'], index=i, key=f"match_{i}_team1_round_{tournament_state['current_round']}")
            
            with col2:
                team2 = st.selectbox(f"Match {i//2 + 1} - Team 2", tournament_state['teams'], index=i + 1, key=f"match_{i}_team2_round_{tournament_state['current_round']}")
            
            matches.append({
                'round': tournament_state['current_round'],
                'team1': team1,
                'team2': team2,
                'score1': 0,
                'score2': 0
            })

    if st.button("Next"):
        if is_decided:
            all_teams = set(tournament_state['teams'])
            selected_teams = {team for match in matches if match['round'] == tournament_state['current_round'] for team in [match['team1'], match['team2']]}

            if selected_teams != all_teams:
                tournament_state['validation_error'] = True
            else:
                tournament_state['validation_error'] = False
                tournament_state['matches'] += matches  # Append to the list of all matches
                next_page()
        else:
            if tournament_state['current_round'] == 1:
                # First round: Shuffle the teams and match 1 vs 2, 3 vs 4, etc.
                shuffled_teams = tournament_state['teams'].copy()
                random.shuffle(shuffled_teams)
                new_matches = []
                for i in range(0, len(shuffled_teams), 2):
                    new_matches.append({
                        'round': 1,
                        'team1': shuffled_teams[i],
                        'team2': shuffled_teams[i + 1],
                        'score1': 0,
                        'score2': 0
                    })
            else:
                # Subsequent rounds: Use past matches and leaderboard
                leaderboard_df = compute_leaderboard(tournament_state['matches'])
                past_matches = tournament_state['matches']
                new_matches = decide_matchup(tournament_state['teams'], round_num=tournament_state['current_round'], past_matches=past_matches, leaderboard=leaderboard_df)
            
            tournament_state['matches'] += new_matches  # Append the new matches
            next_page()

    if tournament_state['validation_error']:
        st.error("Each team must be selected exactly once.")

# Page 3: Show Matchups and Leaderboard
def page_3():
    st.title(f"Tournament Round {tournament_state['current_round']} - Matchups and Leaderboard")
    st.header("Matchups and Leaderboard")

    if 'matches' not in tournament_state or not tournament_state['matches']:
        st.error("No matches have been set up. Please go back to set up matches.")
        return

    # Show matchups for the current round
    st.subheader(f"Round {tournament_state['current_round']} Matchups")
    for match in tournament_state['matches']:
        if match['round'] == tournament_state['current_round']:
            st.markdown(f"**{match['team1']} vs {match['team2']}**")

    # Compute and display the leaderboard
    st.subheader("Leaderboard")
    leaderboard_df = compute_leaderboard(tournament_state['matches'])
    st.table(leaderboard_df)

    if st.button("Next"):
        next_page()

# Page 4: Input Scores for Matches
def page_4():
    st.title(f"Tournament Round {tournament_state['current_round']} - Input Scores")
    st.header("Input Scores")

    if 'matches' not in tournament_state or not tournament_state['matches']:
        st.error("No matches have been set up. Please go back to set up matches.")
        return

    st.subheader("Enter the scores for each match:")
    
    for i, match in enumerate(tournament_state['matches']):
        if match['round'] == tournament_state['current_round']:
            st.markdown(f"### {match['team1']} vs {match['team2']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                match['score1'] = st.number_input(f"Score for {match['team1']}", min_value=0, key=f"score1_{i}_round_{tournament_state['current_round']}", value=match['score1'])
            
            with col2:
                match['score2'] = st.number_input(f"Score for {match['team2']}", min_value=0, key=f"score2_{i}_round_{tournament_state['current_round']}", value=match['score2'])

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Simulate Random Scores"):
            for match in tournament_state['matches']:
                if match['round'] == tournament_state['current_round']:
                    match['score1'] = random.choices(range(6), weights=[5, 10, 7, 3, 2, 1])[0]
                    match['score2'] = random.choices(range(6), weights=[5, 10, 7, 3, 2, 1])[0]
            st.rerun()

    with col2:
        if st.button("Next"):
            next_round()

# Main logic to handle page transitions
if tournament_state['page'] == 1:
    page_1()
elif tournament_state['page'] == 2:
    page_2()
elif tournament_state['page'] == 3:
    page_3()
elif tournament_state['page'] == 4:
    page_4()
elif tournament_state['page'] == 5:
    st.title("Tournament Completed")
    st.header("Final Leaderboard")
    leaderboard_df = compute_leaderboard(tournament_state['matches'])
    st.table(leaderboard_df)