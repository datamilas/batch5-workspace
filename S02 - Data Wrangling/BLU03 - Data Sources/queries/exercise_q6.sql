select t.id as tid, t.short_name , max(m.away_team_goal)
from team_attributes ta 
	inner join team t ON ta.team_id =t.id
	inner join "match" m on m.away_team_id = t.id
where defencepressure>60
group by t.id
having count(t.id)>25 and avg(home_team_goal)>1
order by short_name









