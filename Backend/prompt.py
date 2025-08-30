
# def create_advice_prompt(label, confidence, gps, weather, soil, lang):
#     """
#     Create a dynamic advice prompt for an AI agronomist.

#     Parameters:
#         label (str): Crop or disease name
#         confidence (float): Confidence level of detection
#         gps (str): Location or GPS coordinates
#         weather (str): Current weather information
#         soil (str): Soil type or properties
#         lang (str): Language preference for response

#     Returns:
#         str: Formatted advice prompt
#     """
#     prompt = f"""
# You are an experienced agronomist speaking to a smallholder farmer.

# Crop/Disease: {label}
# Confidence Level: {confidence}
# Location/GPS: {gps}
# Weather Conditions: {weather}
# Soil Type/Properties: {soil}
# Language Preference: {lang}

# Provide advice in simple, practical terms suitable for a smallholder farmer.
# """
#     return prompt

# # Example usage:
# advice_prompt = create_advice_prompt(
#     label="Wheat Rust",
#     confidence=0.92,
#     gps="28.6139° N, 77.2090° E",
#     weather="Sunny, 30°C",
#     soil="Loamy, pH 6.5",
#     lang="Hindi"
# )

# print(advice_prompt)



def create_advice_prompt_multiple(crops, city,gps, weather,  lang):
    """
    Create a dynamic advice prompt for multiple crop recommendations.

    Parameters:
        crops (list): List of crop names (str) suitable for the region
        gps (str): Location or GPS coordinates
        weather (str): Current weather information
        
        lang (str): Language preference for response

    Returns:
        str: Formatted advice prompt
    """
    crops_list = ", ".join(crops)
    
    prompt = f"""
You are an experienced agronomist advising smallholder farmers.
City:{city}
Region/GPS: {gps}
Weather Conditions: {weather}

Language Preference: {lang}

Available crop options for this region: {crops_list}.

You MUST provide a specific recommendation by:
1. CHOOSING ONE BEST CROP from the list: {crops_list}
2. Explaining WHY this crop is the best choice for these conditions
3. Providing SPECIFIC expected yield per acre/hectare
4. Giving APPROXIMATE profit margin percentages
5. Listing sustainability benefits and best practices

Format your response clearly with:
- RECOMMENDED CROP: [specific crop name]
- REASON: [why it's best for this location/soil/weather]
- EXPECTED YIELD: [specific numbers]
- PROFIT MARGIN: [percentage or range]
- SUSTAINABILITY: [environmental benefits]
- BEST PRACTICES: [actionable farming tips]

Be specific with numbers and actionable advice for smallholder farmers.
 Avoid text formating like bold and italic 
     Dont use markdown , give simple text 
"""
    return prompt

def create_user_question_prompt(user_question, gps, weather, city=None, lang="English"):
    return f"""
    You are an expert agronomist helping a small farmer.

    Location: {city} ({gps})
    Weather: {weather}
    

    Question: {user_question}

     Answer in {lang}.
    
     
     
     Avoid text formating like bold and italic 
     Dont use markdown , give simple text 
     dont use "*"

    """
