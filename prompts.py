def prompt_fs_syn_rank_01(dataset_name, user_his_text, candidate_text_order, recall_budget, demo_elems):
    if dataset_name in ['ml-1m', 'ml-01m']:

        demo_prompt = """

--------------------------       
Demonstration Example {}:

The User's Movie Profile:
- Watched Movies: {}

The User's Potential Matches:
- Candidate Movies: {}

Based on the user's watched movies, please rank the candidate movies \
that align closely with the user's preferences. 
- You ONLY rank the given Candidate Movies.
- You DO NOT generate movies from Watched Movies.

Present your response in the format below:
1. [Top Recommendation (Candidate Movie)]
2. [2nd Recommendation (Candidate Movie)]
...
20. [20th Recommendation (Candidate Movie)]

Answer:
{}
            """

        demo_elem_str = ""
        for demo_i, demo_elem in enumerate(demo_elems):
            # print (demo_elem)
            # print ('===')
            input_1 = [str(j) + '. ' + demo_elem[0][j] for j in range(len(demo_elem[0]))]
            input_2 = [str(j) + '. ' + demo_elem[1][j] for j in range(len(demo_elem[1]))]
            # input_3 = [str(j) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))]
            demo_predict = '\n'.join([str(j + 1) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))])
            demo_prompt_str = demo_prompt.format(demo_i, input_1, input_2, demo_predict)
            demo_elem_str += demo_prompt_str

        # assert 1==0

        prompt_icl = f"""
Demonstration Examples:

{demo_elem_str}

------------------

Learn from the above demonstration examples to solve the following test example
Test example:

John's Movie Profile:
- Watched Movies: {user_his_text}

John's Potential Matches:
- Candidate Movies: {candidate_text_order}

Based on John's watched movies, please rank the candidate movies \
that align closely with John's preferences. 
- You ONLY rank the given Candidate Movies.
- You DO NOT generate movies from Watched Movies.

Present your response in the format below:
1. [Top Recommendation (Candidate Movie)]
2. [2nd Recommendation (Candidate Movie)]
...
{recall_budget}. [{recall_budget}th Recommendation (Candidate Movie)]

Answer:

            """

    elif dataset_name == "Games":
        demo_prompt = """
User's Game Product Purchase History:
- Previously Purchased Game Products (in order of purchase): {}

Potential Game Product Recommendations:
- List of Candidate Game Products for Consideration: {}

Considering the user's past game product purchases, please prioritize the game products from the provided list that best match the user's preferences. Ensure that:
- You ONLY rank the given candidate game products.
- You DO NOT generate game products outside of the listed candidate game products.

Present your response in the format below:
1. [Top Recommendation]
2. [2nd Recommendation]
...

{}

"""

        demo_elem_str = ""
        for demo_i, demo_elem in enumerate(demo_elems):
            # print (demo_elem[2])
            input_1 = [str(j) + '. ' + demo_elem[0][j] for j in range(len(demo_elem[0]))]
            input_2 = [str(j) + '. ' + demo_elem[1][j] for j in range(len(demo_elem[1]))]
            # input_3 = [str(j) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))]
            demo_predict = '\n'.join([str(j + 1) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))])
            demo_prompt_str = demo_prompt.format(input_1, input_2, demo_predict)
            demo_elem_str += demo_prompt_str

        prompt_icl = f"""
Examples:
{demo_elem_str}


Learn from the above examples to solve the following test example
Test example:

User's Game Product Purchase History:
- Previously Purchased Game Products (in order of purchase): {user_his_text}

Potential Game Product Recommendations:
- List of Candidate Game Products for Consideration: {candidate_text_order}

Considering the user's past game product purchases, please prioritize the game products from the provided list that best match the user's preferences. Ensure that:
- You ONLY rank the given candidate game products.
- You DO NOT generate game products outside of the listed candidate game products.

Format your recommendations as follows:
1. [Top Recommendation]
2. [2nd Recommendation]
...
{recall_budget}. [{recall_budget}th Recommendation]
            """

    elif dataset_name == "lastfm":
        demo_prompt = """
User's Previously Listened Music Artists:
- Recently Listened Music Artists (in sequential order): {}

Candidate Music Artists for Recommendation:
- Candidate Music Artists: {}

Given the user's listening history, please arrange the candidate music artists in order of relevance to the user's preferences. It is important to:
- Rank ONLY the music artists listed in the candidates.
- Avoid introducing any music artists not included in the candidate list.

Recommendations should be formatted as follows:
1. [Most Recommended Artist]
2. [Second Most Recommended Artist]
...

{}

            """

        demo_elem_str = ""
        for demo_i, demo_elem in enumerate(demo_elems):
            # print (demo_elem[2])
            input_1 = [str(j) + '. ' + demo_elem[0][j] for j in range(len(demo_elem[0]))]
            input_2 = [str(j) + '. ' + demo_elem[1][j] for j in range(len(demo_elem[1]))]
            # input_3 = [str(j) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))]
            demo_predict = '\n'.join([str(j + 1) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))])
            demo_prompt_str = demo_prompt.format(input_1, input_2, demo_predict)
            demo_elem_str += demo_prompt_str

        prompt_icl = f"""
Examples:
{demo_elem_str}


Learn from the above examples to solve the following test example
Test example:

User's Previously Listened Music Artists:
- Recently Listened Music Artists (in sequential order): {user_his_text}

Candidate Music Artists for Recommendation:
- Candidate Music Artists: {candidate_text_order}

Given the user's listening history, please arrange the candidate music artists in order of relevance to the user's preferences. It is important to:
- Rank ONLY the music artists listed in the candidates.
- Avoid introducing any music artists not included in the candidate list.

Recommendations should be formatted as follows:
1. [Most Recommended Artist]
2. [Second Most Recommended Artist]
...
{recall_budget}. [{recall_budget}th Recommended Artist]
"""

    else:
        raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
    return prompt_icl



