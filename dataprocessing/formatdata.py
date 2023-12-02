import csv

comma = ","

# Prompt the user for the label (good or bad)
def get_label():
  success = False

  while (success != True):
    label = input("Please enter the type of domains you are adding (good/bad): ").lower()
    if (label == 'good' or label == 'bad'):
      success = True
      return label
    else:
      print("ERROR! The domains must be 'good' or 'bad'\n")

# Set the label to the user input
label = get_label()

# Open the CSV file
with open('./dataprocessing/incomingdata.csv', mode ='r', encoding="utf-8") as fileInput:
  # Open the CSV file
  with open('./dataset/testdataset.csv', mode ='a', newline='', encoding="utf-8") as fileOutput:
    # open reader to CSV file
    csvRead = csv.reader(fileInput)
    # open writer to CSV file
    csvWrite = csv.writer(fileOutput, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
  
    # Change the contents
    for line in csvRead:
      parameters = []
      # Split the url into sections
      urlSplit = line[0].split(".")
      # Get the number of sections
      numOfSections = len(urlSplit)
      # Get the TLD and its length
      tld = urlSplit[len(urlSplit) - 1]
      tldLength = len(tld)
      # Get the domain and its length
      domain = urlSplit[len(urlSplit) - 2]
      domainLength = len(domain)

      # Add the parameters to the list
      parameters.append(label + comma + str(numOfSections) + comma + tld + comma 
                        + str(tldLength) + comma + domain + comma + str(domainLength) + comma + line[0])
      
      # Write the list to the CSV file
      csvWrite.writerow(parameters)

