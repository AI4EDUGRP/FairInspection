from pyecharts.charts import Pie, Grid
import pyecharts.options as opts
from pyecharts.globals import ThemeType

## Demographic distributions
def demo_distribution(Gender,gender_num,race,race_num):
    pieGender = (
        Pie()
        .add(
            "",
            [list(gender) for gender in zip(Gender, gender_num)],
            radius=["10%", "65%"],
            center=["25%", "60%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Student distributions in gender",
                                     pos_left='2%',
                                     ),
            legend_opts=opts.LegendOpts(pos_left='2%',pos_top='6%')
        )
    )

    pieRace = (
        Pie()
        .add(
            "",
            [list(h) for h in zip(race, race_num)],
            radius=["10%", "65%"],
            center=["75%", "60%"],
            rosetype="radius",
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Student distributions in race",
                                     pos_right='5%'),
            legend_opts=opts.LegendOpts(pos_right='2%',pos_top='5%',pos_left='55%')
        )
    )

    gridDemo=Grid(init_opts=opts.InitOpts(width='1040px',theme=ThemeType.DARK))
    gridDemo.add(pieGender,
         grid_opts=opts.GridOpts(pos_bottom='5%')
        )
    gridDemo.add(pieRace,
        grid_opts=opts.GridOpts(pos_bottom='5%')
        )
    
    return gridDemo