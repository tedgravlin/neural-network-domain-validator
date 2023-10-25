import csv

# Open the CSV file
with open('blocklist.csv', mode ='r') as fileInput:
  # Open the CSV file
  with open('stripped-blocklist.csv', mode ='w', newline='') as fileOutput:
    # open reader to CSV file
    csvRead = csv.reader(fileInput)
    # open writer to CSV file
    csvWrite = csv.writer(fileOutput, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
  
    # Change the contents
    for line in csvRead:
      parameters = []

      # Strip || and ^ from domains 
      parameters.append(line[0][2:-1])
      
      # Write the list to the CSV file
      csvWrite.writerow(parameters)

