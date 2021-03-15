def recipes_from_FoodList(test_labels):
    import requests
    import json

    # 'd3b10f5b15164d13ba2f5e53b9c25a32'
    API_KEY = "4374be7cc066468a800c54923186e141"

    fridge = test_labels  # list(items1)

    num = 4  # number of results to return

    # TODO: potentially unsafe against injection attack
    fridge = ",+".join(fridge)
    # print(fridge)
    # API endpoint to get recipe data
    request1 = 'https://api.spoonacular.com/recipes/findByIngredients?ingredients='
    request1 += fridge + '&number=' + \
        str(num) + '&ranking=2' + '&apiKey=' + API_KEY
    # perform search for recipes with specified input ingredients
    response1 = requests.get(request1)

    # check status code
    if response1.status_code != 200:
        #print('Response Status Code:', response1.status_code)
        quit()

    recipes = response1.json()
    # print(recipes)

    titles = []
    images = []
    Tot_missedIng = []
    Tot_usedIng = []
    # for item in result:
    for recipe in recipes:
        titles.append(recipe['title'])
        images.append(recipe['image'])

        missedIng = []
        usedIng = []

        if recipe['missedIngredientCount'] == 0:
            missedIng.append('None')
        else:
            missedIng = [recipe['missedIngredients'][i]['name']
                         for i in range(0, recipe['missedIngredientCount'])]

        if recipe['usedIngredientCount'] == 0:
            missedIng.append('None')
        else:
            usedIng = [recipe['usedIngredients'][j]['name']
                       for j in range(0, recipe['usedIngredientCount'])]
        Tot_missedIng.append(missedIng)
        Tot_usedIng.append(usedIng)

    #title = recipe['title']
    # print(titles)

    # The spoonacular api outputs titles of recipe suggestions for specified input ingredients
    # It does not output the actual recipe though

    # So we use the google api to search the title name and get the recipe link

    from googleapiclient.discovery import build
    my_api_key = 'AIzaSyDfsCsCcudeIITL6Qb8xKaV6p2uBWRgjmA'
    my_cse_id = "d7bb816fcf044610b"

    def google_search(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res

    results = []
    for title in titles:
        results.append(google_search(title, my_api_key, my_cse_id))

    links = []
    for i in range(0, num):
        links.append(results[i]['items'][0]['link'])  # desired recipe link

    return titles, links, images, Tot_missedIng, Tot_usedIng
