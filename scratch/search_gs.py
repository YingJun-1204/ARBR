import os

def search_text(directory, pattern):
    results = []
    for root, dirs, files in os.walk(directory):
        # Skip pycache and hidden folders
        if '__pycache__' in root or '.git' in root or '.vscode' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                results.append((path, i, line.strip()))
                except Exception as e:
                    pass
    return results

if __name__ == '__main__':
    matches = search_text('.', 'SRSEventSplattingResidual')
    for path, line_no, content in matches:
        print(f"{path}:{line_no}: {content}")
