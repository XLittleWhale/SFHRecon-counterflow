import ast
from pathlib import Path
from counterflow.paths import ROOT
from counterflow.forward.config import Config

def test_no_legacy_source_imports_or_absolute_workspace_paths():
    for path in (ROOT/'src').rglob('*.py'):
        text=path.read_text(encoding='utf8')
        assert 'sys.path' not in text,path
        assert 'F:/' not in text and 'F:\\' not in text,path
        assert 'state_learning' not in text,path
        ast.parse(text)

def test_temperature_dependent_line_density():
    c=Config(temperature_K=2.0,vn_mean_m_s=.004)
    expected=(c.alpha_v/c.beta_v*(c.rho_n_kg_m3+c.rho_s_kg_m3)/c.rho_s_kg_m3*c.vn_mean_m_s)**2
    assert abs(c.line_density_m2/expected-1)<1e-12
