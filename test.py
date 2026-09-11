import importlib.metadata

try:
  top_levels = importlib.metadata.files('openni-python3')
  for file in top_levels or []:
    if 'top_level.txt' in file.path:
      print(file.read_text())
except Exception as e:
  print('Could not find metadata:', e)
