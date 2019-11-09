import React from "react";
import WindowedSelect from "react-windowed-select";

export default class MultiSelect extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      selectedList: []
    };
    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(selectedList) {
    this.setState({ selectedList });
    console.log(`Selected list:`, selectedList);
  }
  render() {
    const { selectedList } = this.state;
    return (
      <WindowedSelect
        options={this.props.options}
        onChange={this.handleChange}
        value={selectedList}
        isMulti
        closeMenuOnSelect={false}
      />
    );
  }
}


