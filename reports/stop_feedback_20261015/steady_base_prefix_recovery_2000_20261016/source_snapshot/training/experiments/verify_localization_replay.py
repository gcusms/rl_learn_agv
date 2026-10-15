"""Check the actual replay controls in a disposable headless Chrome session.

Uses websocket-client, available in the system Python on the experiment host.
Does not open, modify or terminate the user's existing browser session.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import websocket


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();profile=tempfile.TemporaryDirectory(prefix='rl-replay-check-')
    proc=None;ws=None;errors=[];sequence=0
    try:
        proc=subprocess.Popen(['google-chrome','--headless=new','--disable-gpu','--no-first-run',
            '--no-default-browser-check','--remote-debugging-port=0','--remote-allow-origins=*',
            '--window-size=1240,1100','--user-data-dir='+profile.name,(root/'index.html').as_uri()],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        portfile=Path(profile.name)/'DevToolsActivePort';deadline=time.monotonic()+25
        while not portfile.exists():
            if time.monotonic()>deadline or proc.poll() is not None:raise RuntimeError('Headless browser did not start')
            time.sleep(.1)
        port=portfile.read_text().splitlines()[0]
        pages=json.load(urllib.request.urlopen('http://127.0.0.1:'+port+'/json'))
        page=next(page for page in pages if page.get('type')=='page')
        ws=websocket.create_connection(page['webSocketDebuggerUrl'],timeout=30)
        def call(method,params=None):
            nonlocal sequence
            sequence+=1;identity=sequence
            ws.send(json.dumps(dict(id=identity,method=method,params=params or {})))
            while True:
                response=json.loads(ws.recv())
                if response.get('method')=='Runtime.exceptionThrown':errors.append(response['params'])
                if response.get('id')==identity:
                    if 'error' in response:raise RuntimeError(str(response['error']))
                    return response.get('result',{})
        def evaluate(expression):
            value=call('Runtime.evaluate',dict(expression=expression,returnByValue=True))
            if 'exceptionDetails' in value:raise RuntimeError(str(value['exceptionDetails']))
            return value['result'].get('value')
        call('Runtime.enable');call('Page.enable');deadline=time.monotonic()+35
        while not evaluate("typeof data !== 'undefined' && document.getElementById('case').options.length>0"):
            if time.monotonic()>deadline:raise RuntimeError('Replay did not initialize')
            time.sleep(.2)
        time.sleep(.4)
        checks=dict(total_cases=evaluate('data.cases.length'),
            select_options=evaluate("document.getElementById('case').options.length"))
        assert checks['total_cases']==checks['select_options']
        for tag,expression in [('known',"data.cases.findIndex(c=>c.category==='navigation_regression')"),
            ('imu',"data.cases.findIndex(c=>c.mode==='known'&&c.category.includes('IMU'))"),
            ('online',"data.cases.findIndex(c=>c.id.startsWith('imu_online_4_'))")]:
            index=evaluate(expression);assert index>=0
            evaluate("document.getElementById('case').value="+str(index)+
                ";document.getElementById('case').dispatchEvent(new Event('change'));"+
                "document.getElementById('time').value=document.getElementById('time').max;"+
                "document.getElementById('time').dispatchEvent(new Event('input'))")
            time.sleep(.3)
            checks[tag]=dict(result_visible=bool(evaluate("document.getElementById('result').textContent.length>0")),
                map_visible=evaluate("document.getElementById('map').checked"))
            assert checks[tag]['result_visible']
            if tag=='online':assert not checks[tag]['map_visible']
        evaluate("document.getElementById('reset').click();document.getElementById('play').click()")
        time.sleep(.5);checks['play_advances']=evaluate("Number(document.getElementById('time').value)>0")
        assert checks['play_advances']
        evaluate("document.getElementById('play').click();document.getElementById('case').value="+
            "data.cases.findIndex(c=>c.category==='navigation_regression');"+
            "document.getElementById('case').dispatchEvent(new Event('change'));"+
            "document.getElementById('time').value=50;document.getElementById('time').dispatchEvent(new Event('input'))")
        time.sleep(.3);size=call('Page.getLayoutMetrics')['cssContentSize']
        shot=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=True,
            clip=dict(x=0,y=0,width=min(size['width'],2000),height=min(size['height'],4096),scale=1)))
        (root/'REPLAY_SCREENSHOT.png').write_bytes(base64.b64decode(shot['data']))
        checks['javascript_exceptions']=len(errors);assert not errors
        checks['completed']=True;(root/'REPLAY_UI_CHECKS.json').write_text(json.dumps(checks,indent=2))
        print(json.dumps(checks,indent=2))
    finally:
        if ws:ws.close()
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        profile.cleanup()


if __name__=='__main__':main()
