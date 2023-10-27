import csv

comma = ","
#label = "good"
label = "bad"

# Open the CSV file
with open('stripped-blocklist.csv', mode ='r') as fileInput:
  # Open the CSV file
  with open('dataset.csv', mode ='a', newline='') as fileOutput:
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

