select p.name, p.height, p.weight, pa.sprint_speed , pa.acceleration , pa.shot_power 
from player_attributes pa 
	inner join player p on p.id = pa.player_id
where pa.reactions > 85 and pa.potential >= 90
order by pa.positioning desc 