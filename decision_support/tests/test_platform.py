import unittest
from copy import deepcopy
from decision_support.engine import DemoProvider, CONFIGS, SETTINGS, evaluate
from decision_support.server import app, REPLAY

class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.time=[1790280000.]
        self.p=DemoProvider(lambda:self.time[0])
    def advance(self,t):
        self.time[0]+=t
        return self.p.snapshot()
    def test_command_requires_feedback(self):
        s=self.p.snapshot()
        executing=s['applied']['config_id']
        self.assertEqual(s['decision']['status'],'Recommended')
        self.assertIsNone(s['decision']['applied_at'])
        self.assertEqual(self.advance(4)['decision']['status'],'Sent')
        self.assertEqual(self.p.applied,executing)
        self.assertEqual(self.advance(4)['decision']['status'],'Acknowledged')
        self.assertEqual(self.p.applied,executing)
        s=self.advance(3)
        self.assertEqual(s['decision']['status'],'Applied')
        self.assertEqual(s['applied']['config_id'],s['decision']['recommended']['config_id'])
        self.assertEqual(len(s['decision']['feedback']),5)
    def test_unacknowledged_command_keeps_executing_configuration(self):
        executing=self.p.applied
        self.p.action('await_ack')
        s=self.advance(12)
        self.assertEqual(s['decision']['status'],'Sent')
        self.assertEqual(s['applied']['config_id'],executing)
        self.assertIsNone(s['decision']['applied_at'])
    def test_demo_intervals_include_prior_inputs_and_changes(self):
        s=self.p.snapshot()
        prior={d['interval']:d for d in s['history'] if d['interval'] in (1,2)}
        self.assertEqual(set(prior),{1,2})
        self.assertTrue(all(d['recommended'] and d['status']=='Applied' for d in prior.values()))
        self.assertEqual((prior[1]['wind'],prior[1]['k']),(prior[2]['wind'],prior[2]['k']))
        self.assertNotEqual((prior[2]['wind'],prior[2]['k']),(s['decision']['wind'],s['decision']['k']))
        self.assertEqual(s['applied']['config_id'],prior[2]['recommended']['config_id'])
        self.assertTrue(all(d['recommended']['ready_s']<3600 for d in prior.values()))
        seen_configs={d['recommended']['config_id'] for d in s['history']}
        for interval in range(4,9):
            s=self.p.action('interval')
            self.assertEqual(s['mission']['interval'],interval)
            self.assertEqual(s['decision']['interval'],interval)
            self.assertLess(s['forecast']['ready_s'],3600)
            self.assertLess(s['decision']['recommended']['ready_s'],3600)
            seen_configs.add(s['decision']['recommended']['config_id'])
        self.assertGreater(len(seen_configs),2)
        self.assertEqual((s['decision']['wind'],s['decision']['level'],s['decision']['k']),('Tailwind','High',3))
    def test_demo_history_survives_a_delayed_refresh(self):
        s=self.advance(5*25)
        self.assertEqual(s['mission']['interval'],8)
        records={d['interval']:d for d in s['history'] if d['segment_id']=='S2'}
        self.assertEqual(set(records),set(range(1,9)))
        self.assertEqual((records[6]['wind'],records[6]['k']),('Sidewind',5))
    def test_position_reassignment_preserves_drone_ids(self):
        original=deepcopy(self.p.assignment)
        self.p.action('swap')
        self.assertEqual(self.p.assignment,original)
        s=self.advance(9)
        self.assertEqual(set(s['applied']['assignment']),{'D1','D2','D3','D4','D5'})
        self.assertEqual(s['applied']['assignment']['D1'],'P4')
        self.assertEqual(s['applied']['assignment']['D4'],'P1')
        self.assertEqual({d['id'] for d in s['drones']},set(original))
    def test_new_conditions_trigger_new_decision(self):
        initial=self.p.snapshot()['decision']
        old=self.p.seq
        self.p.action('wind');self.assertEqual(self.p.seq,old)
        s=self.advance(2)
        self.assertEqual(s['decision']['wind'],'Sidewind')
        self.assertEqual(s['decision']['level'],'High')
        self.assertGreater(self.p.seq,old)
        wind_decision=s['decision']
        self.p.action('pads');s=self.advance(2)
        self.assertEqual(s['decision']['k'],4)
        self.assertEqual(s['decision']['candidate_limit'],SETTINGS['candidate_limits_by_K']['4'])
        self.assertEqual(s['conditions']['capacity'],6)
        self.assertEqual((initial['wind'],initial['level'],initial['k']),('Headwind','Low',3))
        self.assertEqual((wind_decision['wind'],wind_decision['level'],wind_decision['k']),('Sidewind','High',3))
        self.assertEqual([(d['wind'],d['level'],d['k']) for d in s['history'][:3]],
                         [('Sidewind','High',4),('Sidewind','High',3),('Headwind','Low',3)])
    def test_refresh_does_not_trigger_decision(self):
        seq=self.p.seq
        for _ in range(30):self.p.snapshot()
        self.assertEqual(self.p.seq,seq)
    def test_decision_battery_snapshot_stays_at_decision_time(self):
        original=self.p.snapshot()['decision']
        self.assertEqual(original['input_battery_levels']['D1'],78)
        self.p.batteries[0]=65
        current=self.p.snapshot()
        self.assertEqual(current['drones'][0]['battery'],65)
        self.assertEqual(current['decision']['input_battery_levels']['D1'],78)
        self.p.action('missing')
        unavailable=self.advance(2)['decision']
        self.assertIsNone(unavailable['input_battery_levels']['D3'])
    def test_configuration_comparison_uses_one_decision_input(self):
        s=self.p.snapshot();d=s['decision'];comparison=d['comparison']
        self.assertIsNotNone(comparison)
        self.assertEqual(comparison['input_as_of'],d['input_as_of'])
        self.assertEqual(comparison['before']['config_id'],s['applied']['config_id'])
        self.assertEqual(comparison['before']['assignment'],s['applied']['assignment'])
        self.assertEqual(comparison['after']['config_id'],d['recommended']['config_id'])
        self.assertEqual(comparison['after']['assignment'],d['recommended']['assignment'])
        battery=[d['input_battery_levels'][f'D{i}'] for i in range(1,6)]
        remaining=self.p.remaining()
        for label in ('before','after'):
            side=comparison[label]
            cfg=next(c for c in CONFIGS if c['id']==side['config_id'])
            expected=evaluate(cfg,battery,remaining,d['wind'],d['k'],side['assignment'])
            self.assertEqual({key:side[key] for key in ('arrival_s','charging_s','ready_s')},
                             {key:expected[key] for key in ('arrival_s','charging_s','ready_s')})
        self.assertEqual(comparison['saved_s'],comparison['before']['ready_s']-comparison['after']['ready_s'])
    def test_historical_comparison_stays_frozen_after_execution(self):
        d=self.p.snapshot()['decision'];comparison=deepcopy(d['comparison'])
        self.advance(11)
        self.assertEqual(self.p.applied,comparison['after']['config_id'])
        self.p.batteries[0]=60
        self.p.action('wind');self.advance(2)
        historical=next(row for row in self.p.snapshot()['history'] if row['id']==d['id'])
        self.assertEqual(historical['comparison'],comparison)
        self.assertEqual(historical['comparison']['before']['config_id'],comparison['before']['config_id'])
    def test_missing_decision_input_has_no_comparison(self):
        self.p.action('missing');d=self.advance(2)['decision']
        self.assertIsNone(d['recommended'])
        self.assertIsNone(d['comparison'])
    def test_auto_pause_does_not_pause_progress(self):
        self.p.action('auto');seq=self.p.seq
        s=self.advance(20)
        self.assertEqual(s['mission']['interval'],4)
        self.assertEqual(self.p.seq,seq)
        self.assertEqual(s['mission']['status'],'In transit')
    def test_interval_boundary_changes_decision(self):
        s=self.p.action('interval')
        self.assertEqual(s['mission']['interval'],4)
        self.assertEqual(s['decision']['interval_id'],'S2-I4')
    def test_superseded_result_cannot_apply(self):
        first=self.p.current
        self.p.action('swap');new=self.p.current
        self.assertEqual(first['status'],'Superseded')
        s=self.advance(12)
        self.assertEqual(s['decision']['id'],new['id'])
        self.assertEqual(s['applied']['assignment']['D1'],'P4')
        self.assertIsNone(first['applied_at'])
    def test_time_definition_and_charging_partition(self):
        for k in range(1,6):
            f=evaluate(CONFIGS[0],[78,74,81,72,76],125,'Headwind',k,self.p.assignment)
            self.assertAlmostEqual(f['ready_s'],f['arrival_s']+f['charging_s'],places=1)
            self.assertAlmostEqual(f['charging_s'],max(r['end_s'] for r in f['schedule']),delta=.5)
            self.assertLessEqual(max(r['group'] for r in f['schedule']),k)
            for group in range(1,k+1):
                slots=[r for r in f['schedule'] if r['group']==group]
                for a,b in zip(slots,slots[1:]):self.assertEqual(a['end_s'],b['start_s'])
    def test_missing_data_not_imputed(self):
        self.p.action('missing');s=self.advance(2)
        self.assertIsNone(s['drones'][2]['battery'])
        self.assertIsNone(s['forecast'])
        self.assertEqual(s['decision']['status'],'Unavailable')
        self.assertEqual(s['decision']['candidates'],[])
    def test_delay_is_visible(self):
        s=self.p.action('delay')
        self.assertTrue(all(d['quality']=='Delayed' for d in s['drones']))
        self.assertEqual(s['forecast']['quality'],'Delayed inputs')
    def test_arrival_charging_and_next_segment(self):
        self.p.action('arrive');s=self.advance(4)
        self.assertEqual(s['mission']['status'],'Charging')
        self.assertEqual(s['forecast']['arrival_s'],0)
        self.assertEqual(s['forecast']['ready_s'],s['forecast']['charging_s'])
        self.assertEqual(sum(d['charging_state']=='Charging' for d in s['drones']),3)
        s=self.p.action('complete_charge')
        self.assertTrue(all(d['battery']==99 for d in s['drones']))
        self.assertEqual(s['forecast']['ready_s'],0)
        s=self.p.action('next_segment')
        self.assertEqual(s['mission']['segment_id'],'S3')
        self.assertEqual(s['mission']['interval'],1)
    def test_configurations_have_unique_slots_and_nominal_reference(self):
        for c in CONFIGS:
            self.assertEqual(len(c['positions']),5)
            self.assertEqual(len({p['id'] for p in c['positions']}),5)
            a,b=[next(p for p in c['positions'] if p['id']==i) for i in c['spacing_reference']]
            self.assertAlmostEqual(((a['x']-b['x'])**2+(a['y']-b['y'])**2)**.5,c['spacing_m'],delta=.0001)

class ApiTests(unittest.TestCase):
    def setUp(self):self.client=app.test_client()
    def test_static_and_route(self):
        for path in ['/', '/static/app.js','/static/style.css','/static/map-base.svg']:
            with self.client.get(path) as response:self.assertEqual(response.status_code,200)
        route=self.client.get('/api/route').json
        self.assertEqual(len(route['nodes']),5)
        self.assertEqual(len(route['segments']),4)
        self.assertEqual(sum(n['kind']=='Scenario rooftop' for n in route['nodes']),3)
        self.assertIsNone(route['geographic_scale_to_indoor'])
    def test_live_does_not_show_demo_readings(self):
        s=self.client.get('/api/snapshot?mode=live').json
        self.assertEqual(s['mode'],'Live')
        self.assertNotIn('drones',s)
        self.assertNotIn('forecast',s)
        self.assertEqual(self.client.post('/api/demo/action',json={'mode':'Live','action':'wind'}).status_code,409)
    def test_replay_is_fixed_and_labelled(self):
        a=self.client.get('/api/replay/3').json;b=self.client.get('/api/replay/3').json
        self.assertEqual(a,b)
        self.assertEqual(a['mode'],'Replay')
        self.assertEqual(a['source'],'Recorded demo session')
        self.assertEqual(self.client.get('/api/replay/999').status_code,404)
    def test_replay_labels_match_decision_and_mission_states(self):
        expected={
            'Initial recommendation':'Recommended',
            'Initial command sent':'Sent',
            'Initial command acknowledged':'Acknowledged',
            'Initial configuration applied':'Applied',
            'Interval 4 decision calculating':'Calculating',
            'Wind decision calculating':'Calculating',
            'Wind recommendation':'Recommended',
            'Wind command sent':'Sent',
            'Wind command acknowledged':'Acknowledged',
            'Wind configuration applied':'Applied',
            'Position reassignment requested':'Calculating',
            'Position reassignment recommended':'Recommended',
            'Position reassignment sent':'Sent',
            'Position reassignment acknowledged':'Acknowledged',
            'New assignments applied':'Applied',
            'Interval 5 decision calculating':'Calculating',
            'Pad decision calculating':'Calculating',
            'Pad recommendation':'Recommended',
            'Pad command sent':'Sent',
            'Pad command acknowledged':'Acknowledged',
            'Pad configuration applied':'Applied',
        }
        frames={frame['label']:frame['snapshot'] for frame in REPLAY}
        self.assertEqual(len(frames),len(REPLAY))
        for label,status in expected.items():
            self.assertEqual(frames[label]['decision']['status'],status,label)
        self.assertEqual(frames['Arrived at rooftop node']['mission']['status'],'At node')
        self.assertEqual(frames['Charging starts']['mission']['status'],'Charging')
        self.assertEqual(frames['Ready at node']['mission']['status'],'Ready at node')
    def test_replay_applies_configuration_and_assignment_changes(self):
        frames={frame['label']:frame['snapshot'] for frame in REPLAY}
        initial=frames['Initial recommendation']['applied']
        first=frames['Initial configuration applied']['applied']
        wind=frames['Wind configuration applied']['applied']
        swap_pending=frames['Position reassignment requested']['applied']
        swap=frames['New assignments applied']['applied']
        pads=frames['Pad configuration applied']['applied']
        self.assertNotEqual(initial['config_id'],first['config_id'])
        self.assertNotEqual(first['config_id'],wind['config_id'])
        self.assertEqual(swap_pending['assignment'],wind['assignment'])
        self.assertEqual(swap['config_id'],wind['config_id'])
        self.assertNotEqual(swap['assignment'],swap_pending['assignment'])
        self.assertEqual((swap['assignment']['D1'],swap['assignment']['D4']),('P4','P1'))
        self.assertNotEqual(pads['config_id'],swap['config_id'])
        for label in ('Initial configuration applied','Wind configuration applied',
                      'New assignments applied','Pad configuration applied'):
            snapshot=frames[label]
            self.assertEqual(snapshot['applied']['config_id'],snapshot['decision']['recommended']['config_id'])
            self.assertEqual(snapshot['applied']['assignment'],snapshot['decision']['recommended']['assignment'])
    def test_paper_demo_wind_only_interval_change_saves_four_minutes_25_seconds(self):
        before=REPLAY[3]['snapshot']
        after=REPLAY[10]['snapshot']
        comparison=after['decision']['comparison']
        self.assertEqual((before['mission']['segment_id'],before['mission']['interval']),('S2',3))
        self.assertEqual((after['mission']['segment_id'],after['mission']['interval']),('S2',4))
        self.assertEqual((before['conditions']['wind'],before['conditions']['level']),('Headwind','Low'))
        self.assertEqual((after['conditions']['wind'],after['conditions']['level']),('Sidewind','Low'))
        self.assertEqual((before['conditions']['k'],after['conditions']['k']),(3,3))
        self.assertEqual((before['decision']['k'],after['decision']['k']),(3,3))
        self.assertEqual(before['applied']['config_id'],'vee-50')
        self.assertEqual(after['applied']['config_id'],'diamond-75')
        self.assertEqual(comparison['before']['config_id'],before['applied']['config_id'])
        self.assertEqual(comparison['after']['config_id'],after['applied']['config_id'])
        self.assertEqual(comparison['saved_s'],265)
        self.assertEqual(comparison['before']['ready_s']-comparison['after']['ready_s'],265)
        self.assertEqual(comparison['quality'],'Demo estimate')
    def test_static_figure_intervals_are_consistent_and_isolated(self):
        from decision_support.server import provider
        original=(provider.seq,provider.applied,deepcopy(provider.assignment))
        before=self.client.get('/api/demo/figure/3').json
        after=self.client.get('/api/demo/figure/4').json
        self.assertEqual(before,self.client.get('/api/demo/figure/3').json)
        self.assertEqual(after,self.client.get('/api/demo/figure/4').json)
        self.assertEqual((before['mode'],after['mode']),('Demo','Demo'))
        self.assertEqual((before['mission']['interval'],after['mission']['interval']),(3,4))
        self.assertEqual((before['applied']['config_id'],after['applied']['config_id']),('vee-50','diamond-75'))
        self.assertEqual((before['conditions']['wind'],after['conditions']['wind']),('Headwind','Sidewind'))
        self.assertEqual((before['conditions']['k'],after['conditions']['k']),(3,3))
        self.assertEqual(after['decision']['comparison']['saved_s'],265)
        self.assertEqual(original,(provider.seq,provider.applied,provider.assignment))
        self.assertEqual(self.client.get('/api/demo/figure/5').status_code,404)
    def test_no_hardware_integration(self):self.assertFalse(self.client.get('/api/health').json['hardware_control'])

if __name__=='__main__':unittest.main()
