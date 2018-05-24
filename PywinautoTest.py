# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:12:55 2018

@author: ypy
"""

from pywinauto.application import Application
app = Application(backend="uia").start('notepad.exe')

# describe the window inside Notepad.exe process
#dlg_spec = app['无标题 - 记事本']
dlg_spec = app.window(title='无标题 - 记事本')
#dlg_spec = app.window(best_match='无标题 - 记事本')

# wait till the window is really open
actionable_dlg = dlg_spec.wait('visible')

#dlg_spec = app.window(title='Untitled - Notepad')


print("Mission complete")