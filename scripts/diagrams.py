from graphviz import Digraph
import os

# === Enhanced System Architecture Diagram ===
sys_arch = Digraph('SystemArchitecture', format='png')
sys_arch.attr(rankdir='LR', size='10,5', labelloc='t', fontsize='20', label='System Architecture Diagram')
sys_arch.attr('node', shape='box', style='filled', fillcolor='#E8F0FE', fontname='Helvetica', fontsize='12')

sys_arch.node('Webcam', '📷 Webcam Input')
sys_arch.node('Extractor', '🧵 Frame Extractor')
sys_arch.node('Preprocessor', '🧼 Image Preprocessor')
sys_arch.node('CNN', '🧠 CNN Classifier\n(Deep Learning Model)')
sys_arch.node('Classifier', '🔍 State Classifier\n(Alert / Yawn / Microsleep)')
sys_arch.node('AlertSystem', '🚨 Audio Alert System\n(Beep Trigger)')
sys_arch.node('Interface', '🖥️ Display Interface\n(Real-time Feedback)')

sys_arch.edge('Webcam', 'Extractor', label='video stream')
sys_arch.edge('Extractor', 'Preprocessor', label='frames')
sys_arch.edge('Preprocessor', 'CNN', label='cleaned frames')
sys_arch.edge('CNN', 'Classifier', label='state prediction')
sys_arch.edge('Classifier', 'AlertSystem', label='trigger alert')
sys_arch.edge('Classifier', 'Interface', label='display state')

sys_arch_path = 'enhanced_system_architecture_diagram'
sys_arch.render(sys_arch_path, cleanup=True)
os.startfile(sys_arch_path + '.png')

# === Enhanced Use Case Diagram ===
use_case = Digraph('UseCase', format='png')
use_case.attr(rankdir='LR', labelloc='t', fontsize='20', label='Use Case Diagram')
use_case.attr('node', fontname='Helvetica')

use_case.attr('node', shape='ellipse', style='filled', fillcolor='#FFF2CC')
use_case.node('Driver', '👤 Driver\n(Actor)')

use_case.attr('node', shape='box', style='rounded,filled', fillcolor='#E8F0FE')
use_case.node('Start', '🟢 Start Detection')
use_case.node('Receive', '🔔 Receive Drowsiness Alert')
use_case.node('View', '🖥️ View State on Screen')
use_case.node('End', '🔴 Stop/Exit Application')

use_case.edge('Driver', 'Start')
use_case.edge('Driver', 'View')
use_case.edge('Driver', 'Receive')
use_case.edge('Driver', 'End')

use_case_path = 'enhanced_use_case_diagram'
use_case.render(use_case_path, cleanup=True)
os.startfile(use_case_path + '.png')

# === Enhanced Class Diagram ===
class_diag = Digraph('ClassDiagram', format='png')
class_diag.attr(rankdir='TB', labelloc='t', fontsize='20', label='Class Diagram')
class_diag.attr('node', shape='record', fontname='Helvetica', style='filled', fillcolor='#D1E8E2')

class_diag.node('DataLoader', '''{DataLoader|
- dataset_path: str\\l
+ load_images(): list\\l
+ preprocess_images(): list\\l
}''')

class_diag.node('ModelTrainer', '''{ModelTrainer|
- model: CNN\\l
+ train(): void\\l
+ evaluate(): float\\l
+ save_model(): void\\l
}''')

class_diag.node('InferenceEngine', '''{InferenceEngine|
- webcam_feed: stream\\l
+ infer_live_feed(): str\\l
+ trigger_alert(): void\\l
+ visualize_result(): void\\l
}''')

class_diag.node('MainApp', '''{MainApp|
+ run_pipeline(): void\\l
}''')

class_diag.edge('MainApp', 'DataLoader', label='uses →')
class_diag.edge('MainApp', 'ModelTrainer', label='uses →')
class_diag.edge('MainApp', 'InferenceEngine', label='uses →')

class_diag_path = 'enhanced_class_diagram'
class_diag.render(class_diag_path, cleanup=True)
os.startfile(class_diag_path + '.png')
