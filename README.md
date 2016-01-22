# Process
- Build Classifiers
- Run Server

# Build Classifiers
- Folders: src/classifier
- Write new classifier that inherits src/classifier_controler.Classifier and implement necessary function (see src/classifier/bagofword.py)
- Train the classifier and store it in model/ (see if __main__ in src/classifier_controler.py)

# Run Server
- For releasing, execute src/server.py
- For debugging, execute src/router.py
- The connection settings (url and port) are differnt for releasing and degbugging, please reger to src/test.py
