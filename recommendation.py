def generate_recommendation(cluster):

    recommendations = {

        0: """
        High Investment Farmers:
        - Recommend advanced machinery
        - Suggest premium fertilizers
        - Provide export market insights
        """,

        1: """
        Low Budget Farmers:
        - Provide subsidy schemes
        - Suggest low-cost fertilizers
        - Recommend government support
        """,
         2: """
        Smart Farmers:
        - Recommend AI-based irrigation
        - Suggest modern crop analytics
        - Promote smart farming tools
        """,

        3: """
        Seasonal Farmers:
        - Recommend seasonal crops
        - Suggest weather forecasting tools
        - Provide crop rotation strategies
        """,
         4: """
        Premium Crop Farmers:
        - Suggest export opportunities
        - Recommend premium crop markets
        - Advanced marketing strategies
        """
    }

    return recommendations.get(
        cluster,
        "No recommendation available"
    )