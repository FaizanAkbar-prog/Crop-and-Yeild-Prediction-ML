

from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)


COUNTRIES = [
    'Albania','Algeria','Angola','Argentina','Armenia','Australia','Austria',
    'Azerbaijan','Bahamas','Bahrain','Bangladesh','Belarus','Belgium','Botswana',
    'Brazil','Bulgaria','Burkina Faso','Burundi','Cameroon','Canada',
    'Central African Republic','Chile','Colombia','Croatia','Denmark',
    'Dominican Republic','Ecuador','Egypt','El Salvador','Eritrea','Estonia',
    'Finland','France','Germany','Ghana','Greece','Guatemala','Guinea','Guyana',
    'Haiti','Honduras','Hungary','India','Indonesia','Iraq','Ireland','Italy',
    'Jamaica','Japan','Kazakhstan','Kenya','Latvia','Lebanon','Lesotho','Libya',
    'Lithuania','Madagascar','Malawi','Malaysia','Mali','Mauritania','Mauritius',
    'Mexico','Montenegro','Morocco','Mozambique','Namibia','Nepal','Netherlands',
    'New Zealand','Nicaragua','Niger','Norway','Pakistan','Papua New Guinea',
    'Peru','Poland','Portugal','Qatar','Romania','Rwanda','Saudi Arabia',
    'Senegal','Slovenia','South Africa','Spain','Sri Lanka','Sudan','Suriname',
    'Sweden','Switzerland','Tajikistan','Thailand','Tunisia','Turkey','Uganda',
    'Ukraine','United Kingdom','Uruguay','Zambia','Zimbabwe'
]

CROPS = [
    'Cassava','Maize','Plantains and others','Potatoes','Rice, paddy',
    'Sorghum','Soybeans','Sweet potatoes','Wheat','Yams'
]

CROP_YIELD_CHART = {
    'Cassava': 150479, 'Maize': 36310, 'Plantains and others': 106041,
    'Potatoes': 199802, 'Rice, paddy': 40730, 'Sorghum': 18636,
    'Soybeans': 16731, 'Sweet potatoes': 119058, 'Wheat': 30116, 'Yams': 114140
}

YEAR_TREND_CHART = {
    1990:66447,1991:66319,1992:66916,1993:67480,1994:68517,1995:69524,
    1996:69889,1997:71160,1998:71476,1999:73896,2000:75376,2001:76587,
    2002:77730,2004:80590,2005:80702,2006:80386,2007:82533,2008:84344,
    2009:85350,2010:86513,2011:88908,2012:88570,2013:90357
}



@app.route("/")
def home():
    return render_template("index.html",
                           countries=COUNTRIES,
                           crops=CROPS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        sample = pd.DataFrame([{
            'Area':                          data['country'],
            'Item':                          data['crop'],
            'Year':                          int(data['year']),
            'average_rain_fall_mm_per_year': float(data['rainfall']),
            'pesticides_tonnes':             float(data['pesticides']),
            'avg_temp':                      float(data['temperature'])
        }])

        prediction = model.predict(sample)[0]

        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/stats")
def stats():
    return jsonify({
        "metrics":         metrics,
        "crop_yield":      CROP_YIELD_CHART,
        "year_trend":      YEAR_TREND_CHART
    })


if __name__ == "__main__":
    app.run()
