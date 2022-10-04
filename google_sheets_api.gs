// Insurance Cross Sell
function onOpen() {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('Customer Rank to Call')
    .addItem('Get Prediction', 'PredictAll')
    .addToUi();  
}

// Production server
host_production = 'insurance-cross-selling.herokuapp.com'

// ----- Helper Function ------
// API call
function ApiCall(data, endpoint){
  var url = 'https://' + host_production + endpoint;
  var payload = JSON.stringify( data );

  var options = {'method': 'POST', 'contentType': 'application/json', 'payload': payload};

  Logger.log(url)
  Logger.log(options)

  var response = UrlFetchApp.fetch(url, options);

  // get response
  var rc = response.getResponseCode();
  var responseText = response.getContentText();

  if ( rc !== 200 ){
    Logger.log( 'Response (%s) %s', rc, responseText );
  }
  else{
    prediction = JSON.parse( responseText );
  }
  return prediction
};

// Cross sell propensity score prediction
function PredictAll(){
  //google sheets parameters
  var ss = SpreadsheetApp.getActiveSheet();
  var ssActive = SpreadsheetApp.getActive();
  var titleColumns = ss.getRange('A1:K1').getValues()[0];
  var lastRow = ss.getLastRow();
  var data = ss.getRange('A2' + ':' + 'K' + lastRow).getValues();
  
  // run over all rows
  for (row in data){
    var json = new Object();

    // run over all columns
    for(var j=0; j < titleColumns.length; j++){
      json[titleColumns[j]] = data[row][j];
    };

    // Json file to send
    var json_send = new Object();
    json_send['id'] = json['id']
    json_send['gender'] = json['gender']
    json_send['age'] =  json['age']
    json_send['driving_license'] = json['driving_license']
    json_send['region_code'] = json['region_code']
    json_send['previously_insured'] = json['previously_insured']
    json_send['vehicle_age'] = json['vehicle_age']
    json_send['vehicle_damage'] = json['vehicle_damage']
    json_send['annual_premium'] = json['annual_premium']
    json_send['policy_sales_channel'] = json['policy_sales_channel']
    json_send['vintage'] = json['vintage']

    // Propensity score
    pred = ApiCall( json_send, '/predict' );

    // Send back to google sheets
    ss.getRange(Number(row) + 2 , 12).setValue(pred[0]['score'])

    //Logger.log(pred[0]['score'])
    //Logger.log(row )
  };

  // Sorts descending by column L (Prediction)
  ssActive.getRange('L:L').activate();
  ssActive.getActiveSheet().sort(12, false);

  // Format 2 decimal places
  ssActive.getActiveRangeList().setNumberFormat('0.00')
};